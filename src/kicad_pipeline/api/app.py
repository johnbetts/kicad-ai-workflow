"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from kicad_pipeline.api.schemas import HealthSchema


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="KiCad AI Pipeline API",
        description="AI-assisted PCB design pipeline: requirements to manufacturing files.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Will be restricted in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from kicad_pipeline.api.router_chat import router as chat_router
    from kicad_pipeline.api.router_components import router as components_router
    from kicad_pipeline.api.router_evidence import router as evidence_router
    from kicad_pipeline.api.router_parts import router as parts_router
    from kicad_pipeline.api.router_pipeline import router as pipeline_router

    app.include_router(pipeline_router, prefix="/api/pipeline", tags=["pipeline"])
    app.include_router(evidence_router, prefix="/api/evidence", tags=["evidence"])
    app.include_router(parts_router, prefix="/api/parts", tags=["parts"])
    app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
    app.include_router(components_router, prefix="/api/components", tags=["components"])

    @app.get("/api/health", response_model=HealthSchema)
    async def health() -> HealthSchema:
        return HealthSchema()

    # Mount output directory for serving board images (must be after routers
    # because StaticFiles is a catch-all for its prefix).
    output_dir = Path("output")
    if output_dir.exists():
        app.mount("/api/boards", StaticFiles(directory=str(output_dir)), name="board-files")

    evidence_dir = Path("data/component_evidence")
    if evidence_dir.exists():
        app.mount(
            "/api/component-evidence",
            StaticFiles(directory=str(evidence_dir)),
            name="component-evidence",
        )

    return app


app = create_app()
