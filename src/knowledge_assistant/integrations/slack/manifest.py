"""Render the tunnel-specific Slack app manifest used after bootstrap installation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SLACK_EVENTS_PATH = "/slack/events"
SLACK_BOT_EVENTS = (
    "agent_session_stopped",
    "app_mention",
    "message.channels",
    "message.groups",
)

# Keep this text-equivalent to the root bootstrap artifact. The contract test prevents
# drift while allowing the installed package to generate a final manifest inside the app container.
BOOTSTRAP_MANIFEST = """\
_metadata:
  major_version: 1
  minor_version: 1

display_information:
  name: QA Agent
  description: Answers Slack questions from an internal knowledge base.
  background_color: "#1D2241"

features:
  agent_view:
    agent_description: Mention this agent in an invited channel to search the knowledge base and receive an answer with source artifact IDs.
  bot_user:
    display_name: QA Agent
    always_online: false

oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - assistant:write
      - channels:history
      - chat:write
      - groups:history

settings:
  org_deploy_enabled: false
  socket_mode_enabled: false
  token_rotation_enabled: false
"""


class PublicBaseUrlError(ValueError):
    """Raised when a URL cannot be used as Slack's public Events API origin."""


def build_slack_request_url(public_base_url: str) -> str:
    """Validate an HTTPS origin and append the one public Slack ingress path."""

    normalized_url = public_base_url.strip()
    if not normalized_url or any(character.isspace() for character in normalized_url):
        raise PublicBaseUrlError("public base URL must not be empty or contain whitespace")

    try:
        parsed_url = urlsplit(normalized_url)
        _ = parsed_url.port
    except ValueError as exc:
        raise PublicBaseUrlError("public base URL contains an invalid host or port") from exc

    if parsed_url.scheme.casefold() != "https":
        raise PublicBaseUrlError("public base URL must use HTTPS")
    if parsed_url.hostname is None or not parsed_url.netloc:
        raise PublicBaseUrlError("public base URL must include a host")
    if parsed_url.username is not None or parsed_url.password is not None:
        raise PublicBaseUrlError("public base URL must not include credentials")
    if parsed_url.query or parsed_url.fragment:
        raise PublicBaseUrlError("public base URL must not include a query string or fragment")
    if parsed_url.path not in {"", "/"}:
        raise PublicBaseUrlError("public base URL must be an origin without a path")
    if parsed_url.hostname.casefold() == "localhost":
        raise PublicBaseUrlError("public base URL must be reachable by Slack, not localhost")
    if parsed_url.netloc.endswith(":"):
        raise PublicBaseUrlError("public base URL contains an invalid port")

    return urlunsplit(("https", parsed_url.netloc, SLACK_EVENTS_PATH, "", ""))


def render_configured_manifest(public_base_url: str) -> str:
    """Return a complete manifest with the runtime Request URL and subscribed bot events."""

    request_url = build_slack_request_url(public_base_url)
    # Preserve manifest order and indentation exactly as Slack expects when concatenating
    # the bootstrap body with per-environment event subscriptions.
    rendered_events = "\n".join(f"      - {event_name}" for event_name in SLACK_BOT_EVENTS)
    event_subscriptions = (
        "  event_subscriptions:\n"
        f"    request_url: {request_url}\n"
        "    bot_events:\n"
        f"{rendered_events}\n"
    )
    return f"{BOOTSTRAP_MANIFEST}{event_subscriptions}"


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the complete Slack app manifest after an HTTPS tunnel URL is available."
        )
    )
    parser.add_argument(
        "public_base_url",
        help="HTTPS public origin, for example https://example.ngrok-free.app",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write UTF-8 YAML to this path instead of printing it to standard output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest generator CLI."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = render_configured_manifest(arguments.public_base_url)
    except PublicBaseUrlError as exc:
        parser.error(str(exc))

    output_path: Path | None = arguments.output
    if output_path is None:
        print(manifest, end="")
        return 0

    output_path.write_text(manifest, encoding="utf-8", newline="\n")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
