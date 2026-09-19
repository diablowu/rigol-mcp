"""F02: documented DS1000Z-E BYTE blocks, calibration and backend compatibility."""

import json

import pyvisa
import pytest

from rigol_mcp import drivers, scope as sc, server as srv
from tests.conftest import FakeScope, make_block


def byte_scope(payload, *, model="DS1202Z-E", preamble=None, channel="CHAN1"):
    if preamble is None:
        preamble = f"0,0,{len(payload)},1,2e-6,-1e-3,2,0.02,5,127"
    scope = FakeScope({
        "*IDN?": f"RIGOL TECHNOLOGIES,{model},FIXTURE,0",
        f":{channel}:DISP?": "1", f":{channel}:SCAL?": "0.5",
        f":{channel}:OFFS?": "0.1", f":{channel}:PROB?": "10",
        ":WAV:PRE?": preamble,
    }, read_buffer=make_block(payload))
    scope.read_termination = "\n"
    return scope


@pytest.mark.parametrize("model", ["DS1202Z-E", "DS1102Z-E"])
@pytest.mark.parametrize("backend", ["@py", "@ivi"])
def test_binary_voltage_and_time_calibration(model, backend, monkeypatch):
    monkeypatch.setattr(sc, "active_backend", lambda: backend)
    scope = byte_scope(bytes([0, 10, 127, 132, 255]), model=model, channel="CHAN2")
    result = sc.get_waveform(scope, "chan2")
    # Independently calculated from byte - 5 - 127, with 20 mV per code.
    assert result["voltages_v"] == pytest.approx([-2.64, -2.44, -0.1, 0, 2.46])
    assert result["times_s"] == pytest.approx([-0.001004, -0.001002, -0.001, -0.000998, -0.000996])
    assert result["vmin_v"] == pytest.approx(-2.64)
    assert result["vmax_v"] == pytest.approx(2.46)
    assert result["vmean_v"] == pytest.approx(-0.544)
    assert result["points"] == 5
    assert ":WAV:FORM BYTE" in scope.written
    assert ":WAV:FORM ASC" not in scope.written
    assert scope.written.index(":WAV:STAR 1") < scope.written.index(":WAV:DATA?")
    assert scope.written.index(":WAV:STOP 1200") < scope.written.index(":WAV:DATA?")
    assert scope.read_termination == "\n"
    assert sc.get_driver(scope).name == "DS1000Z-E"


@pytest.mark.parametrize("backend", ["@py", "@ivi"])
def test_binary_reads_disable_termination_for_embedded_lf(monkeypatch, backend):
    monkeypatch.setattr(sc, "active_backend", lambda: backend)
    scope = byte_scope(bytes(range(256)))
    read_bytes = scope.read_bytes
    read_raw = scope.read_raw
    observed = []

    def read_exact(count):
        observed.append(scope.read_termination)
        assert scope.read_termination is None
        return read_bytes(count)

    def read_message():
        observed.append(scope.read_termination)
        assert scope.read_termination is None
        return read_raw()

    monkeypatch.setattr(scope, "read_bytes", read_exact)
    monkeypatch.setattr(scope, "read_raw", read_message)
    result = sc.get_waveform(scope, "CHAN1")
    assert observed
    assert result["points"] == 256
    assert result["voltages_v"][10] == pytest.approx(-2.44)
    assert scope.read_termination == "\n"


@pytest.mark.parametrize("backend", ["@py", "@ivi"])
def test_termination_restored_when_transfer_times_out(monkeypatch, backend):
    monkeypatch.setattr(sc, "active_backend", lambda: backend)
    scope = byte_scope(b"\x7f")

    def fail(*args):
        assert scope.read_termination is None
        raise pyvisa.errors.VisaIOError(-1073807339)

    monkeypatch.setattr(scope, "read_bytes", fail)
    monkeypatch.setattr(scope, "read_raw", fail)
    with pytest.raises(pyvisa.errors.VisaIOError):
        sc.get_waveform(scope, "CHAN1")
    assert scope.read_termination == "\n"


def test_stale_preamble_refreshes_vertical_and_horizontal_calibration():
    preambles = iter([
        "0,0,3,1,0,0,0,0,0,0",  # No valid calibration before first sweep.
        "0,0,3,1,1e-6,-1e-6,0,0.1,3,127",
    ])
    scope = byte_scope(bytes([120, 130, 140]), preamble=lambda: next(preambles))
    result = sc.get_waveform(scope, "CHAN1")
    assert result["voltages_v"] == pytest.approx([-1, 0, 1])
    assert result["times_s"] == pytest.approx([-1e-6, 0, 1e-6])


def test_empty_payload_poll_refreshes_preamble_even_if_initial_xincrement_is_nonzero(monkeypatch):
    monkeypatch.setattr(sc.time, "sleep", lambda _: None)
    scope = byte_scope(b"", preamble="0,0,0,1,1e-6,0,0,0.02,5,127")
    scope.load(make_block(b"") + make_block(bytes([120, 130, 140])))
    preambles = iter([
        "0,0,0,1,1e-6,0,0,0.02,5,127",
        "0,0,3,1,2e-6,0,0,0.1,3,127",
    ])
    scope.responses[":WAV:PRE?"] = lambda: next(preambles)
    result = sc.get_waveform(scope, "CHAN1")
    assert result["voltages_v"] == pytest.approx([-1, 0, 1])
    assert result["time_increment_s"] == 2e-6
    assert scope.written.count(":WAV:DATA?") == 2


def test_empty_byte_payload_reports_no_data(monkeypatch):
    monkeypatch.setattr(sc, "_WAVEFORM_DATA_RETRY_S", 0)
    scope = byte_scope(b"")
    with pytest.raises(RuntimeError, match="no waveform data"):
        sc.get_waveform(scope, "CHAN1")


@pytest.mark.parametrize("preamble,reason", [
    ("0,0,3,1,1e-6,0,0,0.1,3", "10 fields"),
    ("2,0,3,1,1e-6,0,0,0.1,3,127", "BYTE/NORM"),
    ("0,2,3,1,1e-6,0,0,0.1,3,127", "BYTE/NORM"),
    ("0,0,3,1,1e-6,0,0,nan,3,127", "Non-finite"),
    ("0,0,3,1,1e-6,inf,0,0.1,3,127", "Non-finite"),
    ("0,0,3,1,-1e-6,0,0,0.1,3,127", "Invalid increment"),
    ("0,0,3,1,1e-6,0,0,0,3,127", "Invalid increment"),
    ("0,0,3,1,1e-6,0,0,invalid,3,127", "Invalid numeric"),
    ("0,0,4,1,1e-6,0,0,0.1,3,127", "point mismatch"),
    ("0,0,1201,1,1e-6,0,0,0.1,3,127", "point mismatch"),
])
def test_invalid_byte_preamble_never_returns_misleading_voltages(preamble, reason):
    scope = byte_scope(bytes([120, 130, 140]), preamble=preamble)
    with pytest.raises(ValueError, match=reason):
        sc.get_waveform(scope, "CHAN1")


@pytest.mark.parametrize("backend", ["@py", "@ivi"])
def test_truncated_byte_block_is_rejected(monkeypatch, backend):
    monkeypatch.setattr(sc, "active_backend", lambda: backend)
    scope = byte_scope(bytes([120, 130]))
    # Declared 5 bytes, but only 2 samples and LF arrived. Do not let a 2-point
    # preamble make a truncated 5-byte block appear to be a valid record.
    scope.load(b"#15\x78\x82\n")
    with pytest.raises(ValueError, match="Truncated block"):
        sc.get_waveform(scope, "CHAN1")
    assert scope.read_termination == "\n"


@pytest.mark.parametrize("model", ["DS1054Z", "DHO924S"])
def test_other_families_keep_actual_voltage_ascii(model):
    scope = byte_scope(b"-0.1,0.0,0.1", model=model, preamble="2,0,3,1,1e-6,0,0,0.1,3,127")
    scope.responses[":WAV:DATA?"] = "-0.1,0.0,0.1"
    result = sc.get_waveform(scope, "CHAN1")
    assert result["voltages_v"] == pytest.approx([-0.1, 0, 0.1])
    assert ":WAV:FORM ASC" in scope.written
    assert drivers.driver_for(scope.responses["*IDN?"]).name != "DS1000Z-E"


async def test_mcp_samples_keep_time_and_voltage_output_contract(monkeypatch):
    scope = byte_scope(bytes([120, 130, 140]), preamble="0,0,3,1,1e-6,0,0,0.1,3,127")

    async def call(function, *args, **kwargs):
        return function(scope, *args, **kwargs)

    monkeypatch.setattr(srv, "_call", call)
    result = await srv.call_tool("get_waveform", {"channel": "CHAN1", "raw_data": True})
    samples = json.loads(result[0].text)
    assert samples["voltages_v"] == pytest.approx([-1, 0, 1])
    assert samples["times_s"] == pytest.approx([0, 1e-6, 2e-6])
    assert samples["points"] == 3
