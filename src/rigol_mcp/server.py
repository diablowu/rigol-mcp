"""Rigol DS1000Z MCP server."""

import base64
import json
import math
import os
from datetime import datetime
from pathlib import Path


import asyncio
import time

from dotenv import load_dotenv

# Load configuration from a local .env file (e.g. RIGOL_IP, RIGOL_USB, RIGOL_USB_SERIAL)
# before anything reads os.environ. Existing environment variables (e.g. those passed via
# the MCP client's `env` block) take precedence and are not overridden.
load_dotenv()

import mcp.types as types
import pyvisa
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server

from rigol_mcp.waveform_analysis import describe_waveform as _describe_waveform
from rigol_mcp.scope import (
    get_scope, invalidate_scope, usb_in_use, active_backend,
    screenshot_png,
    get_cursor_mode, set_cursor_mode, set_cursor_positions, get_cursor_values,
    send_raw, check_scpi_error,
    run, stop, single, arm_single, trigger_status, autoscale,
    idn, connection_info, set_driver_from_idn,
    measure, measure_between, MEASURE_ITEMS, MEASURE_ITEMS_TWO_SOURCE,
    get_scope_state, set_channel, set_timebase, set_trigger, get_waveform,
    advertised_capabilities, get_channel_state, get_timebase_state, get_trigger_state,
)

server = Server(
    "rigol-mcp",
    instructions=(
        "Rigol oscilloscope control over SCPI.\n"
        "- Read the scope with the data tools first: measure/measure_between for numeric "
        "readings, get_waveform for trace shape/frequency/amplitude analysis, "
        "get_scope_state for configuration. They return compact structured text that is "
        "far cheaper and easier to reason over than an image.\n"
        "- screenshot is a fallback for genuinely visual checks only (on-screen menus, "
        "cursor placement, confirming what a human sees) — do not use it as the default "
        "way to inspect signals.\n"
        "- Call tools strictly sequentially, never concurrently: all commands share one "
        "instrument connection."
    ),
)

# Serialises all VISA operations — the underlying TCP socket is not thread/async safe.
_scope_lock = asyncio.Lock()
_last_call_time: float = 0.0
_MIN_INTERVAL = 0.1          # 100 ms minimum between SCPI operations
_POST_SCREENSHOT_DELAY = 2.0 # scope needs recovery time after large display transfer
_MAX_ATTEMPTS = 3            # initial try + 2 reconnect-and-retry attempts
_RETRY_BACKOFF = 0.2         # seconds to let a flaky USB link settle before retrying
_DEFAULT_SINGLE_CAPTURE_TIMEOUT_S = 5.0
_DEFAULT_SINGLE_CAPTURE_POLL_S = 0.1
_MIN_SINGLE_CAPTURE_TIMEOUT_S = 0.1
_MAX_SINGLE_CAPTURE_TIMEOUT_S = 60.0
_MIN_SINGLE_CAPTURE_POLL_S = 0.02
_MAX_SINGLE_CAPTURE_POLL_S = 1.0

# Communication faults that warrant a reconnect-and-retry. Other exceptions (e.g. bad
# arguments, SCPI errors) are bugs/usage errors and must propagate unretried.
_RETRYABLE = (pyvisa.errors.VisaIOError, UnicodeDecodeError, OSError)

# send_raw sends arbitrary SCPI and can put the scope in any state, so it is opt-in:
# it is only advertised and accepted when RIGOL_ENABLE_SEND_RAW is set to a truthy value.
_SEND_RAW_ENV = "RIGOL_ENABLE_SEND_RAW"


def _send_raw_enabled() -> bool:
    return os.environ.get(_SEND_RAW_ENV, "").strip().lower() not in ("", "0", "false", "no", "off")


class ActionOutcomeUnknown(RuntimeError):
    """A transport fault occurred while a non-idempotent action may have run."""


async def _wait_for_intercommand_gap() -> None:
    """Apply the shared SCPI pacing rule while the caller owns ``_scope_lock``."""
    elapsed = time.monotonic() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - elapsed)


async def _call_locked(fn, *args, retry: bool, **kwargs):
    """Call ``fn`` while holding ``_scope_lock``.

    Query operations can reconnect and replay after a communication error. Actions
    that may already have changed acquisition/configuration use ``retry=False``: the
    cached session is discarded, but the command is never sent to the scope a second
    time and callers receive an explicit unknown-outcome exception.
    """
    attempts = _MAX_ATTEMPTS if retry else 1
    for attempt in range(1, attempts + 1):
        try:
            return fn(get_scope(), *args, **kwargs)
        except _RETRYABLE as exc:
            invalidate_scope()  # a later query/action must establish a clean session
            if not retry:
                raise ActionOutcomeUnknown(
                    "Transport failed while an action may already have reached the scope; "
                    "the action was not retried and its result is unknown."
                ) from exc
            if attempt == attempts:
                raise
            await asyncio.sleep(_RETRY_BACKOFF)


async def _call(fn, *args, retry: bool = True, **kwargs):
    """Call fn(scope, *args, **kwargs) with the cached connection.

    Serialises concurrent calls via a lock, enforces a minimum inter-command gap, and
    recovers and replays only safe query operations by default. Pass ``retry=False`` for
    commands that may change scope state. Such commands are never replayed after a
    transport error; their session is invalidated and :class:`ActionOutcomeUnknown` is
    raised instead.
    """
    global _last_call_time
    async with _scope_lock:
        await _wait_for_intercommand_gap()
        try:
            return await _call_locked(fn, *args, retry=retry, **kwargs)
        finally:
            _last_call_time = time.monotonic()


async def _observed_trigger_status() -> tuple[str | None, str | None]:
    """Best-effort query after an action outcome is unknown; never replays the action."""
    try:
        return await _call(trigger_status), None
    except Exception as exc:  # the original action outcome remains unknown either way
        return None, f"{type(exc).__name__}: {exc}"


async def _action_once(action: str, fn, *args):
    """Run one non-idempotent action and return a structured unknown outcome on I/O loss."""
    try:
        return {"action": action, "outcome": "completed", "value": await _call(fn, *args, retry=False)}
    except ActionOutcomeUnknown as exc:
        status, recovery_error = await _observed_trigger_status()
        result = {
            "action": action,
            "outcome": "unknown",
            "reason": str(exc),
            "trigger_status_after_error": status,
        }
        if recovery_error:
            result["recovery_error"] = recovery_error
        return result


def _single_capture_timing(arguments: dict) -> tuple[float, float]:
    """Validate bounded single-capture timing arguments."""
    timeout_s = float(arguments.get("timeout_s", _DEFAULT_SINGLE_CAPTURE_TIMEOUT_S))
    poll_s = float(arguments.get("poll_interval_s", _DEFAULT_SINGLE_CAPTURE_POLL_S))
    if not math.isfinite(timeout_s) or not _MIN_SINGLE_CAPTURE_TIMEOUT_S <= timeout_s <= _MAX_SINGLE_CAPTURE_TIMEOUT_S:
        raise ValueError(
            f"timeout_s must be finite and between {_MIN_SINGLE_CAPTURE_TIMEOUT_S:g} and "
            f"{_MAX_SINGLE_CAPTURE_TIMEOUT_S:g} seconds"
        )
    if not math.isfinite(poll_s) or not _MIN_SINGLE_CAPTURE_POLL_S <= poll_s <= _MAX_SINGLE_CAPTURE_POLL_S:
        raise ValueError(
            f"poll_interval_s must be finite and between {_MIN_SINGLE_CAPTURE_POLL_S:g} and "
            f"{_MAX_SINGLE_CAPTURE_POLL_S:g} seconds"
        )
    return timeout_s, poll_s


async def _single_capture(timeout_s: float, poll_s: float) -> dict:
    """Arm once and observe trigger state through a bounded, serialized transaction."""
    global _last_call_time
    statuses: list[str] = []
    async with _scope_lock:
        await _wait_for_intercommand_gap()
        try:
            status_before_arm = await _call_locked(trigger_status, retry=True)
            try:
                await _call_locked(arm_single, retry=False)
            except ActionOutcomeUnknown as exc:
                try:
                    recovered_status = await _call_locked(trigger_status, retry=True)
                except Exception as recovery_exc:
                    recovered_status = None
                    recovery_error = f"{type(recovery_exc).__name__}: {recovery_exc}"
                else:
                    recovery_error = None
                result = {
                    "action": "single_capture",
                    "outcome": "unknown",
                    "reason": str(exc),
                    "status_before_arm": status_before_arm,
                    "trigger_status_after_error": recovered_status,
                    "statuses": statuses,
                }
                if recovery_error:
                    result["recovery_error"] = recovery_error
                return result

            deadline = time.monotonic() + timeout_s
            trigger_observed = False
            while True:
                try:
                    status = await _call_locked(trigger_status, retry=True)
                except _RETRYABLE as exc:
                    return {
                        "action": "single_capture",
                        "outcome": "unknown",
                        "reason": "Transport failed while waiting for the armed acquisition; "
                                  "the single command was not retried.",
                        "status_before_arm": status_before_arm,
                        "trigger_status_after_error": None,
                        "recovery_error": f"{type(exc).__name__}: {exc}",
                        "statuses": statuses,
                    }
                statuses.append(status)
                if status == "TD":
                    trigger_observed = True
                if status == "STOP":
                    return {
                        "action": "single_capture",
                        "outcome": "completed" if trigger_observed else "stopped_without_observed_trigger",
                        "status_before_arm": status_before_arm,
                        "final_status": status,
                        "trigger_observed": trigger_observed,
                        "statuses": statuses,
                    }

                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    try:
                        stop_status = await _call_locked(stop, retry=False)
                    except ActionOutcomeUnknown as exc:
                        return {
                            "action": "single_capture",
                            "outcome": "unknown",
                            "reason": f"Timed out waiting for a trigger; :STOP recovery outcome is unknown: {exc}",
                            "status_before_arm": status_before_arm,
                            "trigger_observed": trigger_observed,
                            "statuses": statuses,
                        }
                    return {
                        "action": "single_capture",
                        "outcome": "timed_out_waiting_for_trigger",
                        "status_before_arm": status_before_arm,
                        "final_status": stop_status,
                        "trigger_observed": trigger_observed,
                        "statuses": statuses,
                    }
                await asyncio.sleep(min(poll_s, remaining_s))
        finally:
            _last_call_time = time.monotonic()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = [
        types.Tool(
            name="screenshot",
            description=(
                "Capture a screenshot of the oscilloscope display. "
                "Returns the image and the absolute path where the PNG was saved. "
                "This is a fallback, not the primary way to read the scope: for numeric "
                "readings use measure/measure_between and for trace data use get_waveform — "
                "they return compact structured text that is faster and cheaper to reason "
                "over than an image. Reach for screenshot only when a genuinely visual check "
                "is needed (on-screen menus/UI state, cursor placement, display rendering, "
                "or confirming what a human sees). "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="idn",
            description=(
                "Identify the instrument and report connection details. "
                "Always returns the connection block (transport, RIGOL_USB/RIGOL_IP env "
                "vars, backend hint, resource string, session state, and the detected "
                "dialect driver — DS1000Z, DHO, …) followed by the scope's *IDN? string. "
                "If the *IDN? query fails, the connection block is still returned with "
                "the error — use it to spot LAN-vs-USB misconfig or an unreachable IP "
                "before assuming the scope itself is the problem. "
                "Call this first to verify connectivity and confirm the correct driver "
                "was selected. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_scope_state",
            description=(
                "Return a snapshot of the scope's current configuration: "
                "active channels (scale, offset, coupling, probe), timebase, and trigger. "
                "Call this at the start of a session to understand the current setup. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="set_channel",
            description=(
                "Configure a channel. Only specified parameters are changed. "
                "channel: CHAN1–CHAN4. "
                "scale_v_div: V/div. offset_v: volts. coupling: AC, DC, or GND. "
                "probe: attenuation ratio (1, 10, 100, …). "
                "Parameter names match get_scope_state output for easy round-tripping. "
                "Returns the resulting channel configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel":    {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]},
                    "display":    {"type": "boolean", "description": "Turn channel on/off"},
                    "scale_v_div": {"type": ["number", "string"], "description": "Vertical scale in V/div"},
                    "offset_v":   {"type": ["number", "string"], "description": "Vertical offset in volts"},
                    "coupling":   {"type": "string", "enum": ["AC", "DC", "GND"]},
                    "probe":      {"type": ["number", "string"], "description": "Probe attenuation ratio (e.g. 1, 10, 100)"},
                },
                "required": ["channel"],
            },
        ),
        types.Tool(
            name="set_timebase",
            description=(
                "Set the horizontal timebase. "
                "scale_s_div: seconds per division (e.g. 0.001 for 1 ms/div). "
                "offset_s: shifts the display window; time_start = offset_s − 6×scale_s_div, time_end = offset_s + 6×scale_s_div. "
                "Trigger (t=0) is always a zero crossing when using edge trigger. "
                "To align the right edge to a zero crossing at time T: set offset_s = T − 6×scale_s_div. "
                "To put the trigger at the left edge of the screen: set offset_s = +6×scale_s_div. "
                "Parameter names match get_scope_state output for easy round-tripping. "
                "Returns the resulting timebase configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scale_s_div": {"type": ["number", "string"], "description": "Time per division in seconds"},
                    "offset_s":    {"type": ["number", "string"], "description": "Trigger offset in seconds"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="set_trigger",
            description=(
                "Configure edge trigger. "
                "source: CHAN1–CHAN4 or EXT. "
                "slope: POS (rising), NEG (falling), or RFAL (either). "
                "level: trigger level in volts. "
                "Returns the resulting trigger configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Trigger source, e.g. CHAN1"},
                    "slope":  {"type": "string", "enum": ["POS", "NEG", "RFAL"]},
                    "level":  {"type": ["number", "string"], "description": "Trigger level in volts"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="measure",
            description=(
                "Query a single-source built-in measurement on a channel. "
                "Preferred over screenshot for reading values — numeric results are "
                "cheaper and easier to analyse than an image. "
                "For stable readings: on DS1000Z, stop acquisition first. "
                "On DHO, keep acquisition running — the DHO measurement engine only populates "
                "item values from live acquisitions; some items (VMAX/VMIN/VTOP/FREQUENCY/…) "
                "return 9.9E37 if first queried on a stopped scope. "
                "channel: CHAN1–CHAN4. "
                "item: VMAX, VMIN, VPP, "
                "VTOP (pulse top flat level, histogram-derived — not the same as VMAX), "
                "VBASE (pulse base flat level — not the same as VMIN), "
                "VAMP (=VTOP−VBASE — not the same as VPP=VMAX−VMIN), "
                "VAVG, VRMS (RMS over screen window), PVRMS (RMS over one period), "
                "VUPPER/VMID/VLOWER (timing thresholds at 90%/50%/10% of VAMP by default), "
                "VARIANCE (statistical variance of voltage samples), "
                "FREQUENCY, PERIOD, PWIDTH, NWIDTH, PDUTY, NDUTY, "
                "RTIME, FTIME, OVERSHOOT, PRESHOOT, "
                "PSLEWRATE, NSLEWRATE (slew rate, V/s), "
                "TVMAX, TVMIN (time position at which VMAX/VMIN occurs), "
                "MAREA (waveform area, V·s over screen window), MPAREA (area per period, V·s), "
                "PPULSES, NPULSES, PEDGES, NEDGES. "
                "A return value of 9.9E37 is the scope's invalid/overflow sentinel — "
                "it means the measurement could not be computed (e.g. FREQUENCY returns 9.9E37 "
                "when the timebase is too narrow to show a complete cycle; widen scale and retry); "
                "such values come back annotated as invalid/overflow. "
                "If the channel's display is OFF it is auto-enabled first (noted in the result). "
                "For delay or phase between two channels use measure_between. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]},
                    "item":    {"type": "string", "description": "Measurement item (e.g. FREQUENCY, VPP, VRMS)"},
                },
                "required": ["channel", "item"],
            },
        ),
        types.Tool(
            name="measure_between",
            description=(
                "Query a two-source delay or phase measurement between two channels. "
                "source1 is the reference channel, source2 is the measured channel. "
                "DS1000Z items: RDELAY (rising-edge delay, seconds), FDELAY (falling-edge delay, seconds), "
                "RPHASE (rising-edge phase, degrees), FPHASE (falling-edge phase, degrees). "
                "DHO series exposes a 4-way matrix: RRDELAY/RFDELAY/FRDELAY/FFDELAY and "
                "RRPHASE/RFPHASE/FRPHASE/FFPHASE (first letter = source1 edge, second = source2 edge). "
                "On DHO the DS1000Z names are auto-mapped to their homogeneous equivalents "
                "(RDELAY→RRDELAY, FDELAY→FFDELAY, RPHASE→RRPHASE, FPHASE→FFPHASE). "
                "For stable readings: on DS1000Z, stop acquisition first. "
                "On DHO, keep acquisition running (see `measure` for details). "
                "A 9.9E37 result is the scope's invalid/overflow sentinel and comes back annotated; "
                "any source channel whose display is OFF is auto-enabled first (noted in the result). "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source1": {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"], "description": "Reference channel"},
                    "source2": {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"], "description": "Measured channel"},
                    "item":    {"type": "string", "enum": [
                        "RDELAY", "FDELAY", "RPHASE", "FPHASE",
                        "RRDELAY", "RFDELAY", "FRDELAY", "FFDELAY",
                        "RRPHASE", "RFPHASE", "FRPHASE", "FFPHASE",
                    ]},
                },
                "required": ["source1", "source2", "item"],
            },
        ),
        types.Tool(
            name="get_waveform",
            description=(
                "Download and analyse the current waveform for a channel (NORM screen buffer, up to ~1000–1200 points depending on scope). "
                "Preferred over screenshot for inspecting the trace — the text analysis is "
                "cheaper and easier to reason over than an image. "
                "Stop or single-trigger the scope first for consistent data. "
                "By default returns a plain-text analysis: signal shape, frequency/period, amplitude, "
                "DC offset, cycle count, and data-quality warnings (e.g. mid-cycle edges, invalid frequency). "
                "Amplitude is judged against the channel's V/div: a trace filling under ~10% of the vertical "
                "screen is flagged as noise floor and its shape/frequency are not reported, and one filling "
                "under ~20% gets a low-amplitude warning (reduce V/div and re-capture for a clean signal). "
                "Set raw_data=true to get the full time/voltage JSON arrays instead. "
                "If the channel's display is OFF it is auto-enabled first (flagged in the warnings). "
                "After reading, act on any warnings — if FREQUENCY would be 9.9E37 widen the timebase; "
                "if edges are not near the DC mean, adjust offset so right edge = N×(period/2) − 6×scale. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel":  {"type": "string", "enum": ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]},
                    "raw_data": {"type": "boolean", "description": "Return raw time/voltage JSON arrays instead of text analysis (default false)"},
                },
                "required": ["channel"],
            },
        ),
        types.Tool(
            name="set_cursors",
            description=(
                "Set cursor mode and/or X positions. "
                "mode: OFF, MANUAL (fixed time positions, reads voltage at those X points), "
                "TRACK (cursors snap to and follow the waveform at the X position). "
                "Omit mode to keep current mode. "
                "ax/bx: cursor A/B time positions in seconds. "
                "Returns the resulting cursor readouts. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["OFF", "MANUAL", "TRACK"]},
                    "ax":   {"type": ["number", "string"], "description": "Cursor A X position in seconds"},
                    "bx":   {"type": ["number", "string"], "description": "Cursor B X position in seconds"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_cursor_values",
            description=(
                "Read current cursor mode and all cursor readouts. "
                "AX_s and BX_s are time positions in seconds. "
                "inv_delta_x is 1/Δt — the frequency between the two cursors. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="send_raw",
            description=(
                "Send an arbitrary SCPI command. "
                "Queries (ending with '?') return the response string; "
                "writes return empty string and auto-check the error queue. "
                "Use as an escape hatch when no dedicated tool covers the operation. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "SCPI command, e.g. ':CHAN1:SCAL?' or ':CHAN1:DISP ON'"},
                },
                "required": ["command"],
            },
        ),
        types.Tool(
            name="check_error",
            description="Query the SCPI error queue. Returns the error if present, or 'No error' if clear. Do not call concurrently with any other rigol tool.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="run",
            description="Start continuous acquisition. Returns trigger status after the command. Do not call concurrently with any other rigol tool.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="stop",
            description=(
                "Stop acquisition and freeze the display. "
                "Use before reading measurements or cursors for stable values. "
                "Returns trigger status after the command. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="single",
            description=(
                "Arm the scope for a single acquisition and return the immediately observed trigger status. "
                "WAIT means the scope is still armed, not that a capture completed. "
                "Use single_capture when a bounded wait and an explicit completion outcome are needed. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="single_capture",
            description=(
                "Arm exactly one single acquisition, then poll trigger status until it stops or the deadline expires. "
                "Returns a structured outcome: completed, timed_out_waiting_for_trigger, "
                "stopped_without_observed_trigger, or unknown after a transport fault. "
                "completed requires observing TD before STOP; stopped_without_observed_trigger "
                "does not prove that the frozen record is new. "
                "The scope is stopped on an untriggered timeout. The SINGLE command is never retried. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout_s": {
                        "type": "number", "minimum": _MIN_SINGLE_CAPTURE_TIMEOUT_S,
                        "maximum": _MAX_SINGLE_CAPTURE_TIMEOUT_S,
                        "description": "Maximum time to wait for a trigger; default 5 seconds.",
                    },
                    "poll_interval_s": {
                        "type": "number", "minimum": _MIN_SINGLE_CAPTURE_POLL_S,
                        "maximum": _MAX_SINGLE_CAPTURE_POLL_S,
                        "description": "Trigger-status polling interval; default 0.1 seconds.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="autoscale",
            description=(
                "Run the scope's auto-setup (timebase, vertical scale, trigger). "
                "Takes a few seconds; call get_scope_state afterwards to see the resulting configuration. "
                "Do not call concurrently with any other rigol tool."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]
    # send_raw is an arbitrary-SCPI escape hatch — only expose it when explicitly enabled.
    if not _send_raw_enabled():
        tools = [t for t in tools if t.name != "send_raw"]
    capabilities = advertised_capabilities()
    channels = list(capabilities.channels)
    channel_range = f"CHAN1–CHAN{capabilities.analog_channels}"
    for tool in tools:
        tool.description = tool.description.replace("CHAN1–CHAN4", channel_range)
        for key, schema in tool.inputSchema.get("properties", {}).items():
            if key in {"channel", "source1", "source2"}:
                schema["enum"] = channels.copy()
            elif tool.name == "set_trigger" and key == "source":
                schema["enum"] = list(capabilities.trigger_sources)
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    global _last_call_time
    if name == "screenshot":
        png_bytes = await _call(screenshot_png)
        # pyvisa-py (@py, used with the WinUSB driver) can leave the USBTMC stream in a
        # state where the command right after a large screenshot transfer intermittently
        # times out; dropping the connection forces a clean USBTMC session on the next
        # call (~1 s, within the recovery window below). NI-VISA (@ivi, native USBTMC
        # driver) recovers on its own and a reconnect there actually disrupts the next
        # command — so reconnect only on @py.
        if usb_in_use() and active_backend() == "@py":
            async with _scope_lock:
                invalidate_scope()
        # Advance the cooldown timestamp so the next _call waits for the scope to recover
        # after the large display transfer before sending further SCPI commands.
        _last_call_time = time.monotonic() + _POST_SCREENSHOT_DELAY

        save_dir = Path(os.environ.get("RIGOL_SCREENSHOT_DIR", "screenshots")).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        filename.write_bytes(png_bytes)

        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        return [
            types.TextContent(type="text", text=f"Saved: {filename}"),
            types.ImageContent(type="image", data=b64, mimeType="image/png"),
        ]

    if name == "idn":
        # connection_info reads env vars + module-level state with no device I/O. It's
        # safe to call after any outcome: on success we show what got selected (driver,
        # resource, backend), on failure we show what was attempted (transport, env vars).
        # That way the diagnostic surfaces config issues (LAN vs USB, wrong IP) that
        # would otherwise be hidden behind an opaque VI_ERROR_TMO timeout.
        try:
            idn_str = await _call(idn)
            # Populate the driver cache from the IDN we already have, so the diagnostic
            # below shows which dialect was selected. Avoids a second *IDN? round-trip
            # that get_driver(scope) would otherwise do on first dialect use.
            before = advertised_capabilities()
            set_driver_from_idn(idn_str)
            if advertised_capabilities() != before:
                try:
                    session = server.request_context.session
                except LookupError:
                    pass  # Direct offline calls have no MCP request context.
                else:
                    await session.send_tool_list_changed()
            info = connection_info()
            diag = "\n".join(f"  {k:18s}: {v}" for k, v in info.items())
            return [types.TextContent(type="text",
                text=f"Connection:\n{diag}\n\nIDN: {idn_str}")]
        except Exception as exc:
            info = connection_info()
            diag = "\n".join(f"  {k:18s}: {v}" for k, v in info.items())
            return [types.TextContent(type="text",
                text=f"Connection:\n{diag}\n\n"
                     f"*IDN? query FAILED: {type(exc).__name__}: {exc}\n\n"
                     "The connection details above show what the server was attempting "
                     "when the query failed — check that transport (USB/LAN), RIGOL_IP, "
                     "and RIGOL_USB match how the scope is actually connected.")]

    if name == "get_scope_state":
        state = await _call(get_scope_state)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    if name == "set_channel":
        def _f(key):
            v = arguments.get(key)
            return float(v) if v is not None else None

        def configure(scope):
            set_channel(
                scope, arguments["channel"], display=arguments.get("display"),
                scale=_f("scale_v_div"), offset=_f("offset_v"),
                coupling=arguments.get("coupling"), probe=_f("probe"),
            )
            return get_channel_state(scope, arguments["channel"])

        state = await _call(configure, retry=False)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    if name == "set_timebase":
        def _f(key):
            v = arguments.get(key)
            return float(v) if v is not None else None

        def configure(scope):
            set_timebase(scope, scale=_f("scale_s_div"), offset=_f("offset_s"))
            return get_timebase_state(scope)

        state = await _call(configure, retry=False)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    if name == "set_trigger":
        level = arguments.get("level")
        def configure(scope):
            set_trigger(
                scope, source=arguments.get("source"), slope=arguments.get("slope"),
                level=float(level) if level is not None else None,
            )
            return get_trigger_state(scope)

        state = await _call(configure, retry=False)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    if name == "measure":
        value = await _call(measure, arguments["channel"], arguments["item"], retry=False)
        return [types.TextContent(
            type="text",
            text=f"{arguments['item']} on {arguments['channel']}: {value}",
        )]

    if name == "measure_between":
        value = await _call(measure_between, arguments["source1"], arguments["source2"], arguments["item"], retry=False)
        return [types.TextContent(
            type="text",
            text=f"{arguments['item']} from {arguments['source1']} to {arguments['source2']}: {value}",
        )]

    if name == "get_waveform":
        data = await _call(get_waveform, arguments["channel"], retry=False)
        if arguments.get("raw_data"):
            return [types.TextContent(type="text", text=json.dumps(data))]
        return [types.TextContent(type="text", text=_describe_waveform(data))]

    if name == "set_cursors":
        mode = arguments.get("mode")
        ax = float(arguments["ax"]) if "ax" in arguments else None
        bx = float(arguments["bx"]) if "bx" in arguments else None
        if mode is not None:
            await _call(set_cursor_mode, mode, retry=False)
        else:
            mode = await _call(get_cursor_mode)
        if mode.upper() != "OFF" and (ax is not None or bx is not None):
            await _call(set_cursor_positions, mode, ax=ax, bx=bx, retry=False)
        values = await _call(get_cursor_values)
        lines = "\n".join(f"{k}: {v}" for k, v in values.items())
        return [types.TextContent(type="text", text=lines)]

    if name == "get_cursor_values":
        values = await _call(get_cursor_values)
        lines = "\n".join(f"{k}: {v}" for k, v in values.items())
        return [types.TextContent(type="text", text=lines)]

    if name == "send_raw":
        if not _send_raw_enabled():
            raise ValueError(
                f"send_raw is disabled. Set {_SEND_RAW_ENV}=1 to enable arbitrary SCPI commands."
            )
        response = await _call(send_raw, arguments["command"], retry=arguments["command"].strip().endswith("?"))
        return [types.TextContent(type="text", text=response or "(no response)")]

    if name == "check_error":
        err = await _call(check_scpi_error)
        return [types.TextContent(type="text", text=err or "No error")]

    if name in ("run", "stop", "single"):
        fn = {"run": run, "stop": stop, "single": single}[name]
        result = await _action_once(name, fn)
        if result["outcome"] == "completed":
            return [types.TextContent(type="text", text=f"trigger status: {result['value']}")]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "single_capture":
        timeout_s, poll_s = _single_capture_timing(arguments)
        result = await _single_capture(timeout_s, poll_s)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "autoscale":
        result = await _action_once("autoscale", autoscale)
        if result["outcome"] != "completed":
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        state = await _call(get_scope_state)
        return [types.TextContent(type="text", text=json.dumps(state, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    asyncio.run(_run())


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(
                notification_options=NotificationOptions(tools_changed=True)
            ),
        )


if __name__ == "__main__":
    main()
