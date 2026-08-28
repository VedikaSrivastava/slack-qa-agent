from pathlib import Path

import pytest

from knowledge_assistant.integrations.slack.manifest import (
    BOOTSTRAP_MANIFEST,
    SLACK_BOT_EVENTS,
    PublicBaseUrlError,
    build_slack_request_url,
    main,
    render_configured_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_MANIFEST_PATH = REPOSITORY_ROOT / "slack-app-manifest.yaml"


def test_checked_in_bootstrap_manifest_matches_generator_contract() -> None:
    checked_in_manifest = BOOTSTRAP_MANIFEST_PATH.read_text(encoding="utf-8")

    assert checked_in_manifest == BOOTSTRAP_MANIFEST
    assert "agent_view:" in checked_in_manifest
    assert "display_name: QA Agent" in checked_in_manifest
    assert "grounded-qa-agent" not in checked_in_manifest
    assert "assistant:write" in checked_in_manifest
    assert "chat:write" in checked_in_manifest
    assert "event_subscriptions:" not in checked_in_manifest
    assert "request_url:" not in checked_in_manifest
    assert "socket_mode_enabled: false" in checked_in_manifest


def test_generated_manifest_has_request_url_and_exact_bot_events() -> None:
    generated_manifest = render_configured_manifest("https://example.ngrok-free.app/")

    assert generated_manifest.startswith(BOOTSTRAP_MANIFEST)
    assert "    request_url: https://example.ngrok-free.app/slack/events\n" in generated_manifest
    assert generated_manifest.count("event_subscriptions:") == 1
    for event_name in SLACK_BOT_EVENTS:
        assert generated_manifest.count(f"      - {event_name}\n") == 1
    rendered_event_lines = generated_manifest.partition("    bot_events:\n")[2].splitlines()
    assert rendered_event_lines == [f"      - {event_name}" for event_name in SLACK_BOT_EVENTS]


@pytest.mark.parametrize(
    "public_base_url",
    [
        "",
        "http://public.example.com",
        "https://localhost",
        "https://user:password@public.example.com",
        "https://public.example.com/base",
        "https://public.example.com?query=value",
        "https://public.example.com#fragment",
        "https://public.example.com:not-a-port",
    ],
)
def test_request_url_rejects_non_public_origin_shapes(public_base_url: str) -> None:
    with pytest.raises(PublicBaseUrlError):
        build_slack_request_url(public_base_url)


def test_cli_prints_complete_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["https://public.example.com"]) == 0

    captured = capsys.readouterr()
    assert captured.out == render_configured_manifest("https://public.example.com")
    assert captured.err == ""


def test_cli_writes_complete_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "slack-app-manifest.generated.yaml"

    assert main(["https://public.example.com", "--output", str(output_path)]) == 0

    assert output_path.read_text(encoding="utf-8") == render_configured_manifest(
        "https://public.example.com"
    )
    captured = capsys.readouterr()
    assert captured.out == f"Wrote {output_path}\n"
    assert captured.err == ""
