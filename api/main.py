"""FastAPI entrypoint and application lifecycle management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.predict import get_model_pipeline, set_model_pipeline
from api.routes import router, MODEL_STATUS_GAUGE

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown lifecycle events."""
    logger.info("Initializing %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    try:
        pipeline = get_model_pipeline()
        MODEL_STATUS_GAUGE.set(1)
        logger.info(
            "ML pipeline loaded successfully (version: %s, threshold: %.2f)",
            pipeline.version,
            pipeline.decision_threshold,
        )
    except Exception as exc:
        MODEL_STATUS_GAUGE.set(0)
        logger.warning(
            "Model artifact could not be loaded at startup: %s. "
            "Inference requests will fail until a model is provided.",
            exc,
        )

    yield

    logger.info("Shutting down application...")
    set_model_pipeline(None)


def create_app() -> FastAPI:
    """Application factory for FastAPI service."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "REST API exposing machine learning predictions for early liver cirrhosis "
            "risk assessment across multi-dataset clinical indicators.\n\n"
            "**Research Notice**: Predictions are for research and investigative use only, "
            "not for clinical diagnostic decisions."
        ),
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom exception handler for readable validation errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = []
        for err in exc.errors():
            field = " -> ".join(str(loc) for loc in err.get("loc", []))
            errors.append(
                {
                    "field": field,
                    "message": err.get("msg"),
                    "type": err.get("type"),
                }
            )
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Input Validation Error",
                "details": errors,
                "disclaimer": settings.RESEARCH_DISCLAIMER,
            },
        )

    # Include routes
    app.include_router(router, prefix="")

    # Setup Prometheus Instrumentation
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health", "/ready"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

    @app.get("/", summary="Root Index", tags=["Health & Status"])
    async def root_index():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "documentation": "/docs",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "model_info": "/model-info",
            "disclaimer": settings.RESEARCH_DISCLAIMER,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
