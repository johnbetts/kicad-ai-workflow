"""Tests for the research & learning agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.optimization.research_agent import (
    ResearchAgenda,
    ResearchAgent,
    ResearchCategory,
    ResearchFinding,
    ResearchStatus,
    ResearchTopic,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb_dir(tmp_path: Path) -> Path:
    """Temporary knowledge base directory."""
    d = tmp_path / "research"
    d.mkdir()
    return d


@pytest.fixture()
def agent(kb_dir: Path) -> ResearchAgent:
    """ResearchAgent pointed at tmp dir."""
    return ResearchAgent(knowledge_dir=kb_dir)


def _make_finding(
    *,
    title: str = "Test finding",
    actionable: bool = False,
    relevance: str = "medium",
) -> ResearchFinding:
    return ResearchFinding(
        title=title,
        summary="Summary text",
        source="https://example.com",
        relevance=relevance,  # type: ignore[arg-type]
        actionable=actionable,
        suggested_change="No change needed",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestResearchFinding:
    def test_frozen(self) -> None:
        f = _make_finding()
        with pytest.raises(AttributeError):
            f.title = "changed"  # type: ignore[misc]


class TestResearchTopic:
    def test_frozen(self) -> None:
        t = ResearchTopic(
            topic_id="res-test",
            category="eda_tools",
            title="Test",
            description="desc",
            status="queued",
            priority="medium",
            created_at="2026-01-01",
            updated_at="2026-01-01",
            findings=(),
            tags=(),
        )
        with pytest.raises(AttributeError):
            t.status = "completed"  # type: ignore[misc]


class TestResearchAgenda:
    def test_queued_property(self) -> None:
        t1 = ResearchTopic(
            topic_id="a", category="c", title="T1", description="",
            status="queued", priority="medium",
            created_at="", updated_at="", findings=(), tags=(),
        )
        t2 = ResearchTopic(
            topic_id="b", category="c", title="T2", description="",
            status="completed", priority="medium",
            created_at="", updated_at="", findings=(), tags=(),
        )
        agenda = ResearchAgenda(topics=(t1, t2), last_updated="")
        assert len(agenda.queued) == 1
        assert agenda.queued[0].topic_id == "a"

    def test_completed_property(self) -> None:
        t = ResearchTopic(
            topic_id="a", category="c", title="T", description="",
            status="completed", priority="medium",
            created_at="", updated_at="", findings=(), tags=(),
        )
        agenda = ResearchAgenda(topics=(t,), last_updated="")
        assert len(agenda.completed) == 1

    def test_needs_followup_property(self) -> None:
        t = ResearchTopic(
            topic_id="a", category="c", title="T", description="",
            status="needs_followup", priority="medium",
            created_at="", updated_at="", findings=(), tags=(),
        )
        agenda = ResearchAgenda(topics=(t,), last_updated="")
        assert len(agenda.needs_followup) == 1


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestResearchAgent:
    def test_load_creates_default_agenda(self, agent: ResearchAgent) -> None:
        agenda = agent.load_agenda()
        assert len(agenda.topics) == 10
        assert all(t.status == "queued" for t in agenda.topics)

    def test_save_and_reload(self, agent: ResearchAgent) -> None:
        agenda = agent.load_agenda()
        assert agent.agenda_path.exists()
        # Reload should give same data
        agenda2 = agent.load_agenda()
        assert len(agenda2.topics) == len(agenda.topics)

    def test_add_topic(self, agent: ResearchAgent) -> None:
        agent.load_agenda()
        tid = agent.add_topic(
            category="eda_tools",
            title="New tool research",
            description="Research a new EDA tool",
            priority="high",
            tags=("automation", "routing"),
        )
        assert tid.startswith("res-")
        agenda = agent.load_agenda()
        assert len(agenda.topics) == 11
        added = next(t for t in agenda.topics if t.topic_id == tid)
        assert added.title == "New tool research"
        assert added.tags == ("automation", "routing")

    def test_get_next_topic_priority_order(
        self, agent: ResearchAgent,
    ) -> None:
        agent.load_agenda()
        topic = agent.get_next_topic()
        assert topic is not None
        # Critical should come first
        assert topic.priority == "critical"

    def test_get_next_topic_empty_returns_none(
        self, kb_dir: Path,
    ) -> None:
        agent = ResearchAgent(knowledge_dir=kb_dir)
        # Create agenda with no queued topics
        agenda = ResearchAgenda(topics=(), last_updated="now")
        agent.save_agenda(agenda)
        assert agent.get_next_topic() is None

    def test_get_next_topic_falls_back_to_followup(
        self, kb_dir: Path,
    ) -> None:
        agent = ResearchAgent(knowledge_dir=kb_dir)
        t = ResearchTopic(
            topic_id="fu", category="c", title="Follow up",
            description="", status="needs_followup", priority="medium",
            created_at="", updated_at="", findings=(), tags=(),
        )
        agent.save_agenda(ResearchAgenda(topics=(t,), last_updated="now"))
        result = agent.get_next_topic()
        assert result is not None
        assert result.topic_id == "fu"

    def test_start_topic(self, agent: ResearchAgent) -> None:
        agenda = agent.load_agenda()
        tid = agenda.topics[0].topic_id
        assert agent.start_topic(tid)
        refreshed = agent.load_agenda()
        updated = next(t for t in refreshed.topics if t.topic_id == tid)
        assert updated.status == "in_progress"

    def test_start_topic_not_found(self, agent: ResearchAgent) -> None:
        agent.load_agenda()
        assert not agent.start_topic("nonexistent")

    def test_complete_topic_with_findings(
        self, agent: ResearchAgent,
    ) -> None:
        agenda = agent.load_agenda()
        tid = agenda.topics[0].topic_id
        agent.start_topic(tid)
        findings = (_make_finding(title="Found something"),)
        assert agent.complete_topic(tid, findings)
        refreshed = agent.load_agenda()
        topic = next(t for t in refreshed.topics if t.topic_id == tid)
        assert topic.status == "completed"
        assert len(topic.findings) == 1
        assert topic.findings[0].title == "Found something"

    def test_complete_topic_needs_followup(
        self, agent: ResearchAgent,
    ) -> None:
        agenda = agent.load_agenda()
        tid = agenda.topics[0].topic_id
        assert agent.complete_topic(tid, (), needs_followup=True)
        refreshed = agent.load_agenda()
        topic = next(t for t in refreshed.topics if t.topic_id == tid)
        assert topic.status == "needs_followup"

    def test_generate_research_prompt(self, agent: ResearchAgent) -> None:
        agenda = agent.load_agenda()
        topic = agenda.topics[0]
        prompt = agent.generate_research_prompt(topic)
        assert "## Research Task:" in prompt
        assert topic.title in prompt
        assert topic.category in prompt
        assert "Output Format" in prompt

    def test_get_research_summary(self, agent: ResearchAgent) -> None:
        agent.load_agenda()
        summary = agent.get_research_summary()
        assert summary["total_topics"] == 10
        assert isinstance(summary["by_status"], dict)
        assert isinstance(summary["by_category"], dict)
        assert summary["total_findings"] == 0

    def test_findings_accumulate(self, agent: ResearchAgent) -> None:
        agenda = agent.load_agenda()
        tid = agenda.topics[0].topic_id
        agent.complete_topic(
            tid,
            (_make_finding(title="First"),),
            needs_followup=True,
        )
        # Complete again with more findings
        agent.complete_topic(
            tid,
            (_make_finding(title="Second"),),
        )
        refreshed = agent.load_agenda()
        topic = next(t for t in refreshed.topics if t.topic_id == tid)
        assert len(topic.findings) == 2

    def test_actionable_findings_filed_as_suggestions(
        self, kb_dir: Path, tmp_path: Path,
    ) -> None:
        from kicad_pipeline.agents.suggestions import SuggestionReporter

        sug_path = tmp_path / "suggestions.jsonl"
        reporter = SuggestionReporter("research", suggestions_path=sug_path)
        agent = ResearchAgent(
            knowledge_dir=kb_dir, suggestion_reporter=reporter,
        )
        agenda = agent.load_agenda()
        tid = agenda.topics[0].topic_id

        findings = (
            _make_finding(title="Actionable", actionable=True),
            _make_finding(title="Not actionable", actionable=False),
        )
        agent.complete_topic(tid, findings)

        # Only actionable finding should become a suggestion
        suggestions = reporter.load_all()
        assert len(suggestions) == 1
        assert suggestions[0].title == "Actionable"

    def test_research_status_enum_values(self) -> None:
        assert ResearchStatus.QUEUED.value == "queued"
        assert ResearchStatus.IN_PROGRESS.value == "in_progress"
        assert ResearchStatus.COMPLETED.value == "completed"
        assert ResearchStatus.NEEDS_FOLLOWUP.value == "needs_followup"
        assert ResearchStatus.ARCHIVED.value == "archived"

    def test_research_category_enum_values(self) -> None:
        assert len(ResearchCategory) == 10
        assert ResearchCategory.KICAD_UPDATES.value == "kicad_updates"

    def test_persistence_survives_reload(
        self, kb_dir: Path,
    ) -> None:
        agent1 = ResearchAgent(knowledge_dir=kb_dir)
        agenda = agent1.load_agenda()
        tid = agenda.topics[0].topic_id
        agent1.start_topic(tid)
        agent1.complete_topic(
            tid, (_make_finding(title="Persisted"),),
        )

        # New agent instance should see the changes
        agent2 = ResearchAgent(knowledge_dir=kb_dir)
        agenda2 = agent2.load_agenda()
        topic = next(t for t in agenda2.topics if t.topic_id == tid)
        assert topic.status == "completed"
        assert topic.findings[0].title == "Persisted"
