"""Immutable channel capabilities, separate from the family's SCPI dialect."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeCapabilities:
    model: str
    analog_channels: int
    trigger_sources: tuple[str, ...]

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(f"CHAN{i}" for i in range(1, self.analog_channels + 1))

    def validate_channel(self, channel: str) -> str:
        normalized = channel.upper()
        if normalized not in self.channels:
            raise ValueError(
                f"Invalid channel {channel!r} for {self.model}. Valid: {self.channels}"
            )
        return normalized


def capabilities_for_model(model: str) -> ScopeCapabilities:
    model = model.strip().upper()
    count = 2 if model in {"DS1202Z-E", "DS1102Z-E"} else 4
    channels = tuple(f"CHAN{i}" for i in range(1, count + 1))
    return ScopeCapabilities(model, count, channels + ("EXT", "AC"))


def capabilities_for_idn(identity: str) -> ScopeCapabilities:
    fields = identity.split(",")
    if len(fields) < 2 or not fields[1].strip():
        raise ValueError(f"Missing model in instrument identity: {identity!r}")
    return capabilities_for_model(fields[1])
