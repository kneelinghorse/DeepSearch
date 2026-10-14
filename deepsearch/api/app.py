"""FastAPI application factory for DeepSearch API."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deepsearch.api.queue import JobQueue
from deepsearch.api.routes import health, jobs, missions
from deepsearch.api.worker import JobWorker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Initialize queue with configurable path
    db_path = os.getenv("DEEPSEARCH_JOBS_DB", "deepsearch_jobs.db")
    queue = JobQueue(db_path=db_path)
    app.state.queue = queue
    logger.info(f"Job queue initialized at {db_path}")

    # Start worker in background
    worker = JobWorker(queue)
    worker_task = asyncio.create_task(worker.start())
    logger.info("Job worker started")

    yield

    # Shutdown worker gracefully
    logger.info("Shutting down job worker...")
    worker._shutdown = True
    try:
        await asyncio.wait_for(worker_task, timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Worker shutdown timed out")
        worker_task.cancel()
    logger.info("Job worker stopped")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="DeepSearch API",
        description="Research mission execution API for TraceLab integration",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS configuration
    allowed_origins = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,https://tracelab.namozine.com"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(health.router, tags=["health"])
    app.include_router(missions.router, tags=["missions"])
    app.include_router(jobs.router, tags=["jobs"])

    return app


# Default app instance for uvicorn
app = create_app()
