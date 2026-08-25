"""ASGI entry point."""

from slack_qa_agent.api.app import create_app

app = create_app()
