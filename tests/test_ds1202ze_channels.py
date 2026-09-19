"""F01 regressions: channel limits and configuration readback, entirely offline."""

import json
from dataclasses import FrozenInstanceError
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from rigol_mcp import scope as sc, server as srv
from tests.conftest import FakeScope, make_block


class ModelScope(FakeScope):
    def __init__(self, model="DS1202Z-E", channels=2):
        super().__init__({
            "*IDN?": f"RIGOL TECHNOLOGIES,{model},FIXTURE,0",
            ":SYSTem:ERRor?": "0",
            ":TIM:SCAL?": "0.001", ":TIM:OFFS?": "0", ":TIM:MODE?": "MAIN",
            ":TRIGger:MODE?": "EDGE", ":TRIGger:STATus?": "STOP",
            ":TRIGger:EDGE:SOURce?": "CHAN1", ":TRIGger:EDGE:SLOPe?": "POS",
            ":TRIGger:EDGE:LEVel?": "0.5", ":AUToscale;*OPC?": "1",
            ":MEASure:ITEM?": "1.0", ":WAV:PRE?": "2,0,3,1,1e-6,0,0,1,0,0",
        }, read_buffer=make_block(b"0,1,0"))
        self.timeout = 30000
        for i in range(1, channels + 1):
            for key, value in {"DISP": "1", "SCAL": "1", "OFFS": "0",
                               "COUP": "DC", "PROB": "10"}.items():
                self.responses[f":CHAN{i}:{key}?"] = value
        self.queries = []

    def query(self, command):
        self.queries.append(command)
        return super().query(command)


@pytest.mark.parametrize("model,count", [
    ("DS1202Z-E", 2), ("DS1102Z-E", 2), ("DS1054Z", 4), ("DHO924S", 4),
])
def test_state_queries_exact_model_channel_count(model, count):
    scope = ModelScope(model, count)
    state = sc.get_scope_state(scope)
    assert list(state["channels"]) == [f"CHAN{i}" for i in range(1, count + 1)]
    assert {cmd.split(":")[1] for cmd in scope.queries if cmd.startswith(":CHAN")} == set(state["channels"])
    assert scope.queries.count("*IDN?") == 1


def test_capabilities_are_immutable_and_reset_on_disconnect():
    scope = ModelScope()
    capabilities = sc.get_capabilities(scope)
    with pytest.raises(FrozenInstanceError):
        capabilities.analog_channels = 4
    assert sc.get_capabilities(scope) is capabilities
    sc.invalidate_scope()
    assert sc.get_capabilities(ModelScope("DS1054Z", 4)).analog_channels == 4


@pytest.mark.parametrize("operation", [
    lambda s, ch: sc.set_channel(s, ch, display=True, scale=1),
    lambda s, ch: sc.measure(s, ch, "VPP"),
    lambda s, ch: sc.get_waveform(s, ch),
    lambda s, ch: sc.ensure_channel_displayed(s, ch),
    lambda s, ch: sc.set_trigger(s, source=ch),
    lambda s, ch: sc.measure_between(s, "CHAN1", ch, "RDELAY"),
    lambda s, ch: sc.measure_between(s, ch, "CHAN1", "RDELAY"),
])
@pytest.mark.parametrize("channel", ["CHAN3", "CHAN4"])
def test_invalid_sources_are_rejected_before_source_io(operation, channel):
    scope = ModelScope()
    scope.responses[":CHAN1:DISP?"] = "0"
    with pytest.raises(ValueError, match="Invalid (channel|trigger source)"):
        operation(scope, channel)
    assert scope.written == []
    assert scope.queries == ["*IDN?"]


def test_valid_channel_and_trigger_sources_still_work():
    scope = ModelScope()
    sc.set_channel(scope, "chan2", scale=1)
    assert sc.measure(scope, "chan2", "VPP") == "1.0"
    assert sc.measure_between(scope, "chan1", "chan2", "RDELAY") == "1.0"
    assert sc.get_waveform(scope, "chan2")["points"] == 3
    for source in ["chan2", "EXT", "AC"]:
        sc.set_trigger(scope, source=source)
    assert ":TRIGger:EDGE:SOURce EXT" in scope.written
    assert ":TRIGger:EDGE:SOURce AC" in scope.written


async def test_tool_schema_uses_model_hint_without_hardware(monkeypatch):
    monkeypatch.setenv("RIGOL_MODEL", "DS1202Z-E")
    tools = {tool.name: tool for tool in await srv.list_tools()}
    for name, keys in {
        "set_channel": ["channel"], "measure": ["channel"],
        "get_waveform": ["channel"], "measure_between": ["source1", "source2"],
    }.items():
        for key in keys:
            assert tools[name].inputSchema["properties"][key]["enum"] == ["CHAN1", "CHAN2"]
    assert tools["set_trigger"].inputSchema["properties"]["source"]["enum"] == ["CHAN1", "CHAN2", "EXT", "AC"]
    assert "CHAN1–CHAN4" not in tools["set_channel"].description
    # A hint only shapes the listing; the real model controls runtime limits.
    assert sc.get_capabilities(ModelScope("DS1054Z", 4)).analog_channels == 4


async def test_idn_refreshes_schema_and_notifies_client(monkeypatch):
    scope = ModelScope()
    monkeypatch.setattr(srv, "get_scope", lambda: scope)
    monkeypatch.setattr(srv, "_MIN_INTERVAL", 0)
    session = SimpleNamespace(send_tool_list_changed=AsyncMock())
    with patch.object(type(srv.server), "request_context", new_callable=PropertyMock,
                      return_value=SimpleNamespace(session=session)):
        await srv.call_tool("idn", {})
    session.send_tool_list_changed.assert_awaited_once()
    tools = {tool.name: tool for tool in await srv.list_tools()}
    assert tools["get_waveform"].inputSchema["properties"]["channel"]["enum"] == ["CHAN1", "CHAN2"]


async def test_server_startup_advertises_tool_list_changes(monkeypatch):
    @asynccontextmanager
    async def transport():
        yield ("read", "write")

    run = AsyncMock()
    monkeypatch.setattr(srv, "stdio_server", transport)
    monkeypatch.setattr(srv.server, "run", run)
    await srv._run()
    run.assert_awaited_once()
    options = run.call_args.args[2]
    assert options.capabilities.tools.listChanged is True


@pytest.mark.parametrize("name,arguments,expected", [
    ("set_channel", {"channel": "CHAN2", "scale_v_div": 1}, "scale_v_div"),
    ("set_timebase", {"scale_s_div": 0.001}, "scale_s_div"),
    ("set_trigger", {"source": "CHAN2", "level": 0.5}, "level_v"),
])
async def test_setters_read_back_only_relevant_fields_in_one_call(monkeypatch, name, arguments, expected):
    scope = ModelScope()
    calls = []

    async def call(function, *args, **kwargs):
        calls.append(function)
        return function(scope, *args, **kwargs)

    monkeypatch.setattr(srv, "_call", call)
    result = await srv.call_tool(name, arguments)
    assert expected in json.loads(result[0].text)
    assert len(calls) == 1
    channel_queries = [cmd for cmd in scope.queries if cmd.startswith(":CHAN")]
    if name == "set_channel":
        assert all(cmd.startswith(":CHAN2:") for cmd in channel_queries)
    else:
        assert channel_queries == []


async def test_autoscale_readback_on_two_channels(monkeypatch):
    scope = ModelScope()

    async def call(function, *args, **kwargs):
        return function(scope, *args, **kwargs)

    monkeypatch.setattr(srv, "_call", call)
    result = await srv.call_tool("autoscale", {})
    assert list(json.loads(result[0].text)["channels"]) == ["CHAN1", "CHAN2"]


def test_four_channel_configuration_is_preserved():
    scope = ModelScope("DS1054Z", 4)
    sc.set_channel(scope, "CHAN4", scale=1)
    assert ":CHAN4:SCAL 1" in scope.written
