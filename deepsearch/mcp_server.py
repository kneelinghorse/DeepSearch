"""FastAPI MCP server exposing DeepSearch capabilities."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="DeepSearch MCP Server", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a basic readiness probe response."""

    return {"status": "ok"}


__all__ = ["app"]
