"""Tests for the suggestion reporting system."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.agents.suggestions import Suggestion, SuggestionReporter


class TestSuggestion:
    def test_frozen(self) -> None:
        s = Suggestion(
            suggestion_id="sug-test",
            agent_id="test",
            title="Test",
            description="desc",
            category="quality",
            priority="medium",
            effort="small",
            affected_module="",
            evidence="",
            status="proposed",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            s.status = "accepted"  # type: ignore[misc]


class TestSuggestionReporter:
    def test_suggest_returns_id(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        sid = reporter.suggest(
            title="Test suggestion",
            description="Description",
            category="quality",
        )
        assert sid.startswith("sug-")
        assert path.exists()

    def test_suggest_appends_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        reporter.suggest(title="First", description="d1", category="quality")
        reporter.suggest(title="Second", description="d2", category="performance")
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["title"] == "First"
        second = json.loads(lines[1])
        assert second["title"] == "Second"

    def test_load_all(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        reporter.suggest(title="A", description="d", category="quality")
        reporter.suggest(title="B", description="d", category="testing")
        all_sug = reporter.load_all()
        assert len(all_sug) == 2
        assert all_sug[0].title == "A"
        assert all_sug[1].title == "B"

    def test_load_all_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        assert reporter.load_all() == ()

    def test_update_status(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        sid = reporter.suggest(title="T", description="d", category="quality")
        assert reporter.update_status(sid, "accepted")
        updated = reporter.load_all()
        assert updated[0].status == "accepted"

    def test_update_status_not_found(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        reporter.suggest(title="T", description="d", category="quality")
        assert not reporter.update_status("nonexistent", "accepted")

    def test_filter_by_status(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        sid1 = reporter.suggest(title="A", description="d", category="quality")
        reporter.suggest(title="B", description="d", category="quality")
        reporter.update_status(sid1, "accepted")
        proposed = reporter.filter_by_status("proposed")
        assert len(proposed) == 1
        assert proposed[0].title == "B"

    def test_filter_by_category(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        reporter.suggest(title="A", description="d", category="quality")
        reporter.suggest(title="B", description="d", category="testing")
        quality = reporter.filter_by_category("quality")
        assert len(quality) == 1
        assert quality[0].title == "A"

    def test_suggestion_has_timestamps(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test-agent", suggestions_path=path)
        reporter.suggest(title="T", description="d", category="quality")
        s = reporter.load_all()[0]
        assert s.created_at
        assert s.updated_at

    def test_multiple_agents_append(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        r1 = SuggestionReporter("agent-1", suggestions_path=path)
        r2 = SuggestionReporter("agent-2", suggestions_path=path)
        r1.suggest(title="From 1", description="d", category="quality")
        r2.suggest(title="From 2", description="d", category="performance")
        all_sug = r1.load_all()
        assert len(all_sug) == 2
        assert all_sug[0].agent_id == "agent-1"
        assert all_sug[1].agent_id == "agent-2"

    def test_priority_and_effort_stored(self, tmp_path: Path) -> None:
        path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("test", suggestions_path=path)
        reporter.suggest(
            title="T",
            description="d",
            category="quality",
            priority="critical",
            effort="large",
        )
        s = reporter.load_all()[0]
        assert s.priority == "critical"
        assert s.effort == "large"
