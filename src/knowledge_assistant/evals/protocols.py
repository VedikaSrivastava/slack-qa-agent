"""Code-reviewed offline experiment protocols for reproducible comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    repetitions: int
    max_concurrency: int


SCREENING_PROTOCOL = ExperimentProtocol(
    name="screening",
    repetitions=1,
    max_concurrency=1,
)
CONFIRMATION_PROTOCOL = ExperimentProtocol(
    name="confirmation",
    repetitions=3,
    max_concurrency=1,
)

EXPERIMENT_PROTOCOLS = {
    protocol.name: protocol
    for protocol in (
        SCREENING_PROTOCOL,
        CONFIRMATION_PROTOCOL,
    )
}


def get_experiment_protocol(name: str) -> ExperimentProtocol:
    try:
        return EXPERIMENT_PROTOCOLS[name]
    except KeyError as exc:
        choices = ", ".join(EXPERIMENT_PROTOCOLS)
        raise ValueError(f"Unknown experiment protocol {name!r}; choose one of: {choices}") from exc
