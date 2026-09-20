"""Application factory.

Shared resources are created once in the lifespan handler and released on shutdown.
Creating an engine or HTTP client per request would waste connections and, on this
hardware, memory.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint

from app.api.routes import auth, health, repositories, review, runs
from app.config.settings import Settings, get_settings
from app.db.session import dispose_engine, init_engine
from app.observability.logging_setup import configure_logging, request_id_var

logger = logging.getLogger(__name__)


def _error_body(code: str, message: str, request_id: str | None) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()

    configure_logging(settings.log_level)
    settings.validate_for_runtime()
    init_engine(settings)

    logger.info(
        "api starting env=%s sandbox=%s db=%s",
        settings.app_env,
        settings.sandbox_backend,
        "sqlite" if settings.is_sqlite else "postgres",
    )

    yield

    await dispose_engine()
    logger.info("api stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Pullsmith",
        version="0.1.0",
        description="Turns a GitHub issue into a reviewed pull request.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Assigns a request id and logs one line per request.

        The id is attached to the log context and echoed in the response so a user can
        report an id and the matching server logs can be found.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error %s %s", request.method, request.url.path)
            raise
        finally:
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                request_id=request.headers.get("X-Request-ID") or request_id_var.get(),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "request_id": request_id_var.get(),
                    "fields": [
                        {"location": list(err["loc"]), "message": err["msg"]}
                        for err in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Never leaks internals.

        The detail is logged server-side with the request id; the client receives a
        generic message so stack traces, queries and paths stay private.
        """
        logger.exception("unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_body(
                code="INTERNAL_ERROR",
                message="An internal error occurred",
                request_id=request_id_var.get(),
            ),
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(repositories.router)
    app.include_router(runs.router)
    app.include_router(review.router)

    return app


app = create_app()

