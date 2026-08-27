"""FastAPI application factory and runtime dependency wiring."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import inngest.fast_api
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.agent.processor import create_question_processor
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE
from knowledge_assistant.application.question_processor import QuestionProcessor
from knowledge_assistant.config import (
    APPLICATION_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_VERSION,
    SlackApplicationSettings,
    get_slack_application_settings,
)
from knowledge_assistant.execution.dispatcher import InngestQuestionDispatcher
from knowledge_assistant.execution.inngest import create_inngest_client, create_question_function
from knowledge_assistant.integrations.slack.app import create_slack_app
from knowledge_assistant.integrations.slack.publisher import SlackPublisher
from knowledge_assistant.observability.logging import (
    bind_run_context,
    clear_run_context,
    configure_logging,
)
from knowledge_assistant.persistence.database import create_database_engine, database_is_ready
from knowledge_assistant.persistence.repositories import PostgresRunLedger


def create_app(settings: SlackApplicationSettings | None = None) -> FastAPI:
    settings = settings or get_slack_application_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            async with create_question_processor(settings, PRODUCTION_PROFILE) as processor:
                app.state.question_processor = processor
                yield
        finally:
            # Engine cleanup must also run when processor startup or shutdown fails.
            await engine.dispose()

    app = FastAPI(title="Slack Q&A Agent", version=APPLICATION_VERSION, lifespan=lifespan)
    inngest_client = create_inngest_client(settings)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        bind_run_context(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            clear_run_context()

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> JSONResponse:
        is_knowledge_database_ready = await anyio.Path(settings.knowledge_db_path).is_file()
        is_postgres_ready = await database_is_ready(engine)
        is_ready = is_knowledge_database_ready and is_postgres_ready
        payload = {
            "status": "ready" if is_ready else "not_ready",
            "knowledge_database": ("available" if is_knowledge_database_ready else "missing"),
            "postgres": "available" if is_postgres_ready else "unavailable",
            "agent": "configured",
            "slack": "configured",
            "inngest": "configured",
        }
        return JSONResponse(payload, status_code=200 if is_ready else 503)

    ledger = PostgresRunLedger(
        engine,
        prompt_version=PROMPT_VERSION,
        retrieval_version=RETRIEVAL_VERSION,
        model_name=PRODUCTION_PROFILE.model_name,
    )
    dispatcher = InngestQuestionDispatcher(inngest_client)
    slack_app = create_slack_app(settings, dispatcher, ledger)
    slack_handler = AsyncSlackRequestHandler(slack_app)
    publisher = SlackPublisher(
        AsyncWebClient(token=settings.slack_bot_token.get_secret_value()), ledger
    )

    @app.post("/slack/events")
    async def slack_events(request: Request) -> Response:
        return await slack_handler.handle(request)

    def processor_provider() -> QuestionProcessor:
        processor: QuestionProcessor = app.state.question_processor
        return processor

    functions: list[Any] = [
        create_question_function(
            inngest_client,
            processor_provider=processor_provider,
            ledger=ledger,
            publisher=publisher,
        )
    ]
    inngest.fast_api.serve(app, inngest_client, functions)
    return app
