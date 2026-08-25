"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

import inngest.fast_api
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from slack_qa_agent.config import Settings, get_settings
from slack_qa_agent.execution.dispatcher import InngestQuestionDispatcher
from slack_qa_agent.execution.inngest import create_inngest_client
from slack_qa_agent.integrations.slack.app import create_slack_app
from slack_qa_agent.observability.logging import configure_logging

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="Slack Q&A Agent", version="0.1.0")
    inngest_client = create_inngest_client(settings)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readiness() -> JSONResponse:
        database_exists = Path(settings.knowledge_db_path).is_file()
        payload = {
            "status": "ready" if database_exists else "not_ready",
            "database": "available" if database_exists else "missing",
            "slack": "configured" if settings.slack_configured else "not_configured",
            "agent": "configured" if settings.agent_configured else "not_configured",
        }
        return JSONResponse(payload, status_code=200 if database_exists else 503)

    if settings.slack_configured:
        dispatcher = InngestQuestionDispatcher(inngest_client)
        slack_app = create_slack_app(settings, dispatcher)
        slack_handler = AsyncSlackRequestHandler(slack_app)

        @app.post("/slack/events")
        async def slack_events(request: Request) -> Response:
            return await slack_handler.handle(request)
    else:
        logger.warning("slack_routes_not_mounted", reason="missing Slack credentials")

    # Functions are added once the QuestionProcessor and LangGraph agent are wired.
    inngest.fast_api.serve(app, inngest_client, [])
    return app
