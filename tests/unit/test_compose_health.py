from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPOSITORY_ROOT / "compose.yaml"


def test_slack_ingress_waits_for_real_inngest_healthcheck() -> None:
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    slack_ingress_block = compose_text.partition("\n  slack-ingress:\n")[2].partition(
        "\n  inngest:\n"
    )[0]
    inngest_block = compose_text.partition("\n  inngest:\n")[2].partition("\nvolumes:\n")[0]

    assert "      inngest:\n        condition: service_healthy" in slack_ingress_block
    assert '      test: ["CMD", "inngest", "alpha", "doctor", "healthcheck"]' in inngest_block
