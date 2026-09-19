"""Unit tests for rigol_mcp.server — the _call retry/recovery wrapper and the
backend-aware screenshot reconnect. VISA access is faked; no hardware involved."""

import json
import time

import pyvisa
import pytest

import rigol_mcp.server as srv


@pytest.fixture(autouse=True)
def fast_and_isolated(monkeypatch):
    """Remove real timing and connection management from _call for unit testing."""
    monkeypatch.setattr(srv, "_RETRY_BACKOFF", 0)
    monkeypatch.setattr(srv, "get_scope", lambda: "FAKE_SCOPE")
    monkeypatch.setattr(srv, "invalidate_scope", lambda: invalidated.append(1))
    # advance the clock baseline so the min-interval gate doesn't sleep by default
    srv._last_call_time = time.monotonic() - 10.0
    invalidated.clear()


invalidated: list = []


def _tmo():
    return pyvisa.errors.VisaIOError(-1073807339)  # VI_ERROR_TMO


async def test_call_returns_result():
    assert await srv._call(lambda scope: f"ok:{scope}") == "ok:FAKE_SCOPE"


async def test_call_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(scope):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _tmo()
        return "recovered"

    assert await srv._call(flaky) == "recovered"
    assert calls["n"] == 3
    assert len(invalidated) == 2          # reconnected before each of the 2 retries


async def test_call_exhausts_attempts_then_raises():
    calls = {"n": 0}

    def always(scope):
        calls["n"] += 1
        raise _tmo()

    with pytest.raises(pyvisa.errors.VisaIOError):
        await srv._call(always)
    assert calls["n"] == srv._MAX_ATTEMPTS
    assert len(invalidated) == srv._MAX_ATTEMPTS  # invalidate after every failed attempt


async def test_call_does_not_replay_non_idempotent_action_after_transport_error():
    calls = {"n": 0}

    def action(scope):
        calls["n"] += 1
        raise _tmo()

    with pytest.raises(srv.ActionOutcomeUnknown, match="not retried"):
        await srv._call(action, retry=False)
    assert calls["n"] == 1
    assert len(invalidated) == 1


async def test_call_does_not_retry_non_communication_errors():
    calls = {"n": 0}

    def bad(scope):
        calls["n"] += 1
        raise ValueError("bad argument")

    with pytest.raises(ValueError):
        await srv._call(bad)
    assert calls["n"] == 1                 # no retry
    assert invalidated == []               # no reconnect


@pytest.mark.parametrize("exc_factory", [
    _tmo,
    lambda: UnicodeDecodeError("utf-8", b"", 0, 1, "boom"),
    lambda: OSError("usb gone"),
])
async def test_call_retries_each_retryable_type(exc_factory):
    calls = {"n": 0}

    def flaky(scope):
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc_factory()
        return "ok"

    assert await srv._call(flaky) == "ok"
    assert calls["n"] == 2


async def test_call_enforces_min_interval(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(srv.asyncio, "sleep", fake_sleep)
    srv._last_call_time = time.monotonic()   # just ran -> next call must wait
    await srv._call(lambda scope: "ok")
    assert slept and slept[0] > 0
    assert slept[0] <= srv._MIN_INTERVAL + 0.01


# --------------------------------------------------------------------------- single-capture transaction

async def test_single_action_reports_unknown_without_sending_a_second_command(monkeypatch):
    writes = {"n": 0}

    def interrupted_single(scope):
        writes["n"] += 1
        raise _tmo()

    monkeypatch.setattr(srv, "single", interrupted_single)
    monkeypatch.setattr(srv, "trigger_status", lambda scope: "WAIT")

    result = await srv.call_tool("single", {})
    payload = json.loads(result[0].text)
    assert writes["n"] == 1
    assert payload["outcome"] == "unknown"
    assert payload["trigger_status_after_error"] == "WAIT"


async def test_single_capture_completes_only_after_observed_trigger_and_stop(monkeypatch):
    arms = {"n": 0}
    statuses = iter(["TD", "WAIT", "TD", "STOP"])

    def arm(scope):
        arms["n"] += 1

    monkeypatch.setattr(srv, "arm_single", arm)
    monkeypatch.setattr(srv, "trigger_status", lambda scope: next(statuses))

    result = await srv._single_capture(timeout_s=0.2, poll_s=0.001)
    assert arms["n"] == 1
    assert result == {
        "action": "single_capture",
        "outcome": "completed",
        "status_before_arm": "TD",
        "final_status": "STOP",
        "trigger_observed": True,
        "statuses": ["WAIT", "TD", "STOP"],
    }


async def test_single_capture_stops_after_an_untriggered_deadline(monkeypatch):
    stops = {"n": 0}

    monkeypatch.setattr(srv, "arm_single", lambda scope: None)
    statuses = iter(["TD"] + ["WAIT"] * 20)
    monkeypatch.setattr(srv, "trigger_status", lambda scope: next(statuses))

    def stop_after_timeout(scope):
        stops["n"] += 1
        return "STOP"

    monkeypatch.setattr(srv, "stop", stop_after_timeout)

    result = await srv._single_capture(timeout_s=0.1, poll_s=0.02)
    assert stops["n"] == 1
    assert result["outcome"] == "timed_out_waiting_for_trigger"
    assert result["status_before_arm"] == "TD"
    assert result["final_status"] == "STOP"
    assert result["trigger_observed"] is False
    assert result["statuses"] and set(result["statuses"]) == {"WAIT"}


async def test_single_capture_reports_unknown_after_arm_transport_error_without_replay(monkeypatch):
    arms = {"n": 0}

    def interrupted_arm(scope):
        arms["n"] += 1
        raise _tmo()

    monkeypatch.setattr(srv, "arm_single", interrupted_arm)
    statuses = iter(["TD", "WAIT"])
    monkeypatch.setattr(srv, "trigger_status", lambda scope: next(statuses))

    result = await srv._single_capture(timeout_s=0.2, poll_s=0.02)
    assert arms["n"] == 1
    assert result["outcome"] == "unknown"
    assert result["status_before_arm"] == "TD"
    assert result["trigger_status_after_error"] == "WAIT"
    assert result["statuses"] == []


@pytest.mark.parametrize("arguments", [
    {"timeout_s": 0},
    {"timeout_s": float("inf")},
    {"poll_interval_s": 0.001},
    {"poll_interval_s": float("nan")},
])
def test_single_capture_timing_rejects_invalid_values(arguments):
    with pytest.raises(ValueError):
        srv._single_capture_timing(arguments)


async def test_single_capture_is_advertised():
    names = {tool.name for tool in await srv.list_tools()}
    assert "single_capture" in names


# --------------------------------------------------------------------------- screenshot reconnect

async def _run_screenshot(monkeypatch, tmp_path, *, usb, backend):
    """Drive call_tool('screenshot') with a faked _call and capture invalidate calls."""
    monkeypatch.setenv("RIGOL_SCREENSHOT_DIR", str(tmp_path))
    png = b"\x89PNG\r\n\x1a\n" + b"fakeimage"

    async def fake_call(fn, *a, **k):
        return png

    calls = {"invalidate": 0}
    monkeypatch.setattr(srv, "_call", fake_call)
    monkeypatch.setattr(srv, "usb_in_use", lambda: usb)
    monkeypatch.setattr(srv, "active_backend", lambda: backend)
    monkeypatch.setattr(srv, "invalidate_scope", lambda: calls.__setitem__("invalidate", calls["invalidate"] + 1))

    result = await srv.call_tool("screenshot", {})
    return result, calls, tmp_path


async def test_screenshot_reconnects_on_py_backend(monkeypatch, tmp_path):
    result, calls, out = await _run_screenshot(monkeypatch, tmp_path, usb=True, backend="@py")
    assert calls["invalidate"] == 1                      # @py: reconnect to reset USBTMC
    assert any(getattr(b, "type", None) == "image" for b in result)
    assert list(out.glob("*.png"))                       # saved to disk


async def test_screenshot_no_reconnect_on_ivi_backend(monkeypatch, tmp_path):
    _, calls, _ = await _run_screenshot(monkeypatch, tmp_path, usb=True, backend="@ivi")
    assert calls["invalidate"] == 0                      # @ivi recovers on its own


async def test_screenshot_no_reconnect_on_lan(monkeypatch, tmp_path):
    _, calls, _ = await _run_screenshot(monkeypatch, tmp_path, usb=False, backend=None)
    assert calls["invalidate"] == 0


# --------------------------------------------------------------------------- send_raw gating

async def test_send_raw_hidden_by_default():
    names = {t.name for t in await srv.list_tools()}
    assert "send_raw" not in names


async def test_send_raw_listed_when_enabled(monkeypatch):
    monkeypatch.setenv("RIGOL_ENABLE_SEND_RAW", "1")
    names = {t.name for t in await srv.list_tools()}
    assert "send_raw" in names


async def test_send_raw_call_rejected_when_disabled():
    with pytest.raises(ValueError, match="send_raw is disabled"):
        await srv.call_tool("send_raw", {"command": ":CHAN1:SCAL?"})


async def test_send_raw_call_works_when_enabled(monkeypatch):
    monkeypatch.setenv("RIGOL_ENABLE_SEND_RAW", "1")

    async def fake_call(fn, *a, **k):
        return "1.000000e+00"

    monkeypatch.setattr(srv, "_call", fake_call)
    result = await srv.call_tool("send_raw", {"command": ":CHAN1:SCAL?"})
    assert result[0].text == "1.000000e+00"
