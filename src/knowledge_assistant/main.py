"""ASGI entry point."""

from knowledge_assistant.api.app import create_app

app = create_app()
