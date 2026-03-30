"""Tests for kicad_pipeline.api endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kicad_pipeline.api.app import app


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    """GET /api/health returns 200 with status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------------------------------------------------------------------------
# Evidence / boards
# ---------------------------------------------------------------------------


def test_list_boards_empty(client: TestClient, tmp_path: Path) -> None:
    """GET /api/evidence/boards returns empty list for empty output dir."""
    resp = client.get(
        "/api/evidence/boards",
        params={"output_dir": str(tmp_path)},
    )
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_boards_nonexistent_dir(client: TestClient, tmp_path: Path) -> None:
    """GET /api/evidence/boards returns empty for nonexistent directory."""
    resp = client.get(
        "/api/evidence/boards",
        params={"output_dir": str(tmp_path / "does_not_exist")},
    )
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Parts search
# ---------------------------------------------------------------------------


def test_parts_search_no_db(client: TestClient) -> None:
    """GET /api/parts/search returns empty when JLCPCB DB not available."""
    resp = client.get("/api/parts/search", params={"q": "100nF"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "100nF"
    assert isinstance(body["results"], list)


def test_parts_search_with_filters(client: TestClient) -> None:
    """GET /api/parts/search accepts filter params without error."""
    resp = client.get(
        "/api/parts/search",
        params={"q": "10k", "basic_only": "true", "in_stock": "true", "limit": "5"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Pipeline stage execution
# ---------------------------------------------------------------------------


def test_invalid_stage_returns_400(client: TestClient) -> None:
    """POST /api/pipeline/run/invalid returns 400."""
    resp = client.post(
        "/api/pipeline/run/invalid",
        json={"board_name": "test"},
        params={"requirements_json": "{}"},
    )
    assert resp.status_code == 400


def test_missing_requirements_returns_400(client: TestClient) -> None:
    """POST /api/pipeline/run/requirements without requirements_json returns 400."""
    resp = client.post(
        "/api/pipeline/run/requirements",
        json={"board_name": "test"},
    )
    # Router raises HTTPException(400) when requirements_json is empty
    assert resp.status_code == 400


def test_invalid_json_returns_400(client: TestClient) -> None:
    """POST /api/pipeline/run/requirements with bad JSON returns 400."""
    resp = client.post(
        "/api/pipeline/run/requirements",
        json={"board_name": "test"},
        params={"requirements_json": "{not valid json}"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_health_response_schema(client: TestClient) -> None:
    """Health endpoint response matches HealthSchema."""
    resp = client.get("/api/health")
    body = resp.json()
    assert "status" in body
    assert "version" in body
    assert "test_count" in body


def test_parts_search_response_schema(client: TestClient) -> None:
    """Parts search response matches PartSearchResponseSchema."""
    resp = client.get("/api/parts/search", params={"q": "resistor"})
    body = resp.json()
    assert "query" in body
    assert "results" in body
    assert "total" in body
    assert isinstance(body["total"], int)
