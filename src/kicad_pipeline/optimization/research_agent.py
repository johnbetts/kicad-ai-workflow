"""Research & learning agent for continuous pipeline improvement.

Systematically researches EDA tools, KiCad updates, JLCPCB specification
changes, routing algorithms, and PCB design best practices.  Logs findings
as structured research entries and proposes pipeline improvements via the
suggestion system.

The agent maintains a knowledge base of research topics and their status,
enabling it to track what has been investigated and what needs follow-up.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kicad_pipeline.agents.suggestions import SuggestionReporter

log = logging.getLogger(__name__)

_DEFAULT_KNOWLEDGE_BASE_DIR = (
    Path.home() / ".claude" / "kicad-agents" / "research"
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResearchStatus(Enum):
    """Lifecycle status of a research topic."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_FOLLOWUP = "needs_followup"
    ARCHIVED = "archived"


class ResearchCategory(Enum):
    """Category of research topic."""

    KICAD_UPDATES = "kicad_updates"
    JLCPCB_SPECS = "jlcpcb_specs"
    ROUTING_ALGORITHMS = "routing_algorithms"
    PLACEMENT_STRATEGIES = "placement_strategies"
    EDA_TOOLS = "eda_tools"
    COMPONENT_LIBRARIES = "component_libraries"
    DESIGN_RULES = "design_rules"
    MANUFACTURING = "manufacturing"
    SIGNAL_INTEGRITY = "signal_integrity"
    THERMAL_MANAGEMENT = "thermal_management"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchFinding:
    """A single finding from a research session."""

    title: str
    summary: str
    source: str
    relevance: Literal["high", "medium", "low"]
    actionable: bool
    suggested_change: str


@dataclass(frozen=True)
class ResearchTopic:
    """A topic queued for or completed by research."""

    topic_id: str
    category: str
    title: str
    description: str
    status: str
    priority: str
    created_at: str
    updated_at: str
    findings: tuple[ResearchFinding, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ResearchAgenda:
    """Prioritised list of research topics."""

    topics: tuple[ResearchTopic, ...]
    last_updated: str

    @property
    def queued(self) -> tuple[ResearchTopic, ...]:
        """Return topics awaiting research."""
        return tuple(t for t in self.topics if t.status == ResearchStatus.QUEUED.value)

    @property
    def completed(self) -> tuple[ResearchTopic, ...]:
        """Return completed topics."""
        return tuple(
            t for t in self.topics if t.status == ResearchStatus.COMPLETED.value
        )

    @property
    def needs_followup(self) -> tuple[ResearchTopic, ...]:
        """Return topics needing follow-up."""
        return tuple(
            t
            for t in self.topics
            if t.status == ResearchStatus.NEEDS_FOLLOWUP.value
        )


# ---------------------------------------------------------------------------
# Default research agenda
# ---------------------------------------------------------------------------

_DEFAULT_TOPICS: tuple[tuple[str, str, str, str], ...] = (
    (
        ResearchCategory.KICAD_UPDATES.value,
        "KiCad 10 S-expression format changes",
        "Track breaking changes in KiCad 10 file formats, new S-expression "
        "nodes, deprecated fields, and version-gated features",
        "critical",
    ),
    (
        ResearchCategory.JLCPCB_SPECS.value,
        "JLCPCB manufacturing capability updates",
        "Monitor changes to JLCPCB minimum trace/space, via sizes, layer "
        "stackup options, and new assembly capabilities",
        "high",
    ),
    (
        ResearchCategory.ROUTING_ALGORITHMS.value,
        "FreeRouting alternatives and improvements",
        "Research alternative open-source autorouters, FreeRouting forks, "
        "and ML-based routing approaches",
        "high",
    ),
    (
        ResearchCategory.PLACEMENT_STRATEGIES.value,
        "ML-based component placement",
        "Investigate machine learning approaches to PCB placement including "
        "reinforcement learning and graph neural networks",
        "medium",
    ),
    (
        ResearchCategory.EDA_TOOLS.value,
        "KiCad Python scripting API",
        "Research KiCad's pcbnew Python API for direct board manipulation, "
        "DRC integration, and zone filling",
        "high",
    ),
    (
        ResearchCategory.COMPONENT_LIBRARIES.value,
        "JLCPCB parts library integration",
        "Monitor JLCPCB parts database updates, new basic parts, and "
        "improved search APIs",
        "medium",
    ),
    (
        ResearchCategory.DESIGN_RULES.value,
        "IPC standards for clearance and creepage",
        "Research IPC-2221B standards for voltage-dependent clearance, "
        "creepage distances, and altitude derating",
        "medium",
    ),
    (
        ResearchCategory.SIGNAL_INTEGRITY.value,
        "High-speed design rule automation",
        "Research automated impedance control, length matching, and "
        "differential pair routing for USB/HDMI/DDR",
        "medium",
    ),
    (
        ResearchCategory.THERMAL_MANAGEMENT.value,
        "Thermal via and copper pour optimization",
        "Research thermal via arrays, copper pour strategies, and "
        "component-level thermal analysis integration",
        "low",
    ),
    (
        ResearchCategory.MANUFACTURING.value,
        "Panelization and multi-board assembly",
        "Research automated panelization for JLCPCB, V-score/tab routing "
        "generation, and fiducial placement",
        "low",
    ),
)


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------


class ResearchAgent:
    """Agent that continuously researches EDA improvements.

    Maintains a knowledge base of topics and findings, and proposes
    pipeline improvements via the suggestion system.

    The agent does NOT perform web searches itself — it structures
    research tasks that a Claude Code agent can execute, then records
    the findings for future reference.
    """

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        suggestion_reporter: SuggestionReporter | None = None,
    ) -> None:
        self._kb_dir = knowledge_dir or _DEFAULT_KNOWLEDGE_BASE_DIR
        self._suggestion_reporter = suggestion_reporter
        self._agenda: ResearchAgenda | None = None

    # -- persistence --------------------------------------------------------

    @property
    def agenda_path(self) -> Path:
        """Path to the research agenda JSON file."""
        return self._kb_dir / "agenda.json"

    def _ensure_dir(self) -> None:
        """Create the knowledge base directory if needed."""
        self._kb_dir.mkdir(parents=True, exist_ok=True)

    def load_agenda(self) -> ResearchAgenda:
        """Load the research agenda from disk, or create defaults."""
        if self.agenda_path.exists():
            data = json.loads(self.agenda_path.read_text(encoding="utf-8"))
            topics: list[ResearchTopic] = []
            for td in data.get("topics", []):
                findings = tuple(
                    ResearchFinding(**f) for f in td.pop("findings", [])
                )
                td["findings"] = findings
                td["tags"] = tuple(td.get("tags", []))
                topics.append(ResearchTopic(**td))
            self._agenda = ResearchAgenda(
                topics=tuple(topics),
                last_updated=data.get("last_updated", ""),
            )
        else:
            self._agenda = self._create_default_agenda()
            self.save_agenda(self._agenda)
        return self._agenda

    def save_agenda(self, agenda: ResearchAgenda) -> None:
        """Persist the research agenda to disk."""
        self._ensure_dir()
        data = {
            "topics": [
                {
                    **asdict(t),
                    "findings": [asdict(f) for f in t.findings],
                    "tags": list(t.tags),
                }
                for t in agenda.topics
            ],
            "last_updated": agenda.last_updated,
        }
        self.agenda_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        self._agenda = agenda

    def _create_default_agenda(self) -> ResearchAgenda:
        """Create initial agenda from default topics."""
        now = datetime.now(tz=timezone.utc).isoformat()
        topics: list[ResearchTopic] = []
        for category, title, description, priority in _DEFAULT_TOPICS:
            topics.append(
                ResearchTopic(
                    topic_id=f"res-{uuid.uuid4().hex[:12]}",
                    category=category,
                    title=title,
                    description=description,
                    status=ResearchStatus.QUEUED.value,
                    priority=priority,
                    created_at=now,
                    updated_at=now,
                    findings=(),
                    tags=(),
                )
            )
        return ResearchAgenda(topics=tuple(topics), last_updated=now)

    # -- topic management ---------------------------------------------------

    def add_topic(
        self,
        category: str,
        title: str,
        description: str,
        priority: Literal["critical", "high", "medium", "low"] = "medium",
        tags: tuple[str, ...] = (),
    ) -> str:
        """Add a new research topic. Returns the topic ID."""
        agenda = self.load_agenda()
        now = datetime.now(tz=timezone.utc).isoformat()
        topic = ResearchTopic(
            topic_id=f"res-{uuid.uuid4().hex[:12]}",
            category=category,
            title=title,
            description=description,
            status=ResearchStatus.QUEUED.value,
            priority=priority,
            created_at=now,
            updated_at=now,
            findings=(),
            tags=tags,
        )
        new_agenda = ResearchAgenda(
            topics=(*agenda.topics, topic),
            last_updated=now,
        )
        self.save_agenda(new_agenda)
        return topic.topic_id

    def get_next_topic(self) -> ResearchTopic | None:
        """Return the highest-priority queued topic, or None."""
        agenda = self.load_agenda()
        queued = agenda.queued
        if not queued:
            # Check for follow-ups
            followups = agenda.needs_followup
            if followups:
                return followups[0]
            return None

        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return min(
            queued,
            key=lambda t: priority_order.get(t.priority, 99),
        )

    def start_topic(self, topic_id: str) -> bool:
        """Mark a topic as in-progress. Returns True if found."""
        return self._update_topic_status(
            topic_id, ResearchStatus.IN_PROGRESS.value
        )

    def complete_topic(
        self,
        topic_id: str,
        findings: tuple[ResearchFinding, ...],
        needs_followup: bool = False,
    ) -> bool:
        """Record findings and mark topic complete or needs_followup."""
        agenda = self.load_agenda()
        now = datetime.now(tz=timezone.utc).isoformat()
        new_status = (
            ResearchStatus.NEEDS_FOLLOWUP.value
            if needs_followup
            else ResearchStatus.COMPLETED.value
        )

        found = False
        new_topics: list[ResearchTopic] = []
        for t in agenda.topics:
            if t.topic_id == topic_id:
                found = True
                new_topics.append(
                    ResearchTopic(
                        topic_id=t.topic_id,
                        category=t.category,
                        title=t.title,
                        description=t.description,
                        status=new_status,
                        priority=t.priority,
                        created_at=t.created_at,
                        updated_at=now,
                        findings=t.findings + findings,
                        tags=t.tags,
                    )
                )
                # File suggestions for actionable findings
                self._file_suggestions_for_findings(t, findings)
            else:
                new_topics.append(t)

        if found:
            self.save_agenda(
                ResearchAgenda(topics=tuple(new_topics), last_updated=now)
            )
        return found

    def _update_topic_status(self, topic_id: str, new_status: str) -> bool:
        """Update a topic's status. Returns True if found."""
        agenda = self.load_agenda()
        now = datetime.now(tz=timezone.utc).isoformat()
        found = False
        new_topics: list[ResearchTopic] = []
        for t in agenda.topics:
            if t.topic_id == topic_id:
                found = True
                new_topics.append(
                    ResearchTopic(
                        topic_id=t.topic_id,
                        category=t.category,
                        title=t.title,
                        description=t.description,
                        status=new_status,
                        priority=t.priority,
                        created_at=t.created_at,
                        updated_at=now,
                        findings=t.findings,
                        tags=t.tags,
                    )
                )
            else:
                new_topics.append(t)
        if found:
            self.save_agenda(
                ResearchAgenda(topics=tuple(new_topics), last_updated=now)
            )
        return found

    # -- suggestion integration ---------------------------------------------

    def _file_suggestions_for_findings(
        self,
        topic: ResearchTopic,
        findings: tuple[ResearchFinding, ...],
    ) -> None:
        """Convert actionable findings into pipeline suggestions."""
        if self._suggestion_reporter is None:
            return

        category_map: dict[str, str] = {
            ResearchCategory.KICAD_UPDATES.value: "architecture",
            ResearchCategory.JLCPCB_SPECS.value: "quality",
            ResearchCategory.ROUTING_ALGORITHMS.value: "performance",
            ResearchCategory.PLACEMENT_STRATEGIES.value: "performance",
            ResearchCategory.EDA_TOOLS.value: "architecture",
            ResearchCategory.COMPONENT_LIBRARIES.value: "quality",
            ResearchCategory.DESIGN_RULES.value: "quality",
            ResearchCategory.SIGNAL_INTEGRITY.value: "quality",
            ResearchCategory.THERMAL_MANAGEMENT.value: "quality",
            ResearchCategory.MANUFACTURING.value: "quality",
        }

        for f in findings:
            if not f.actionable:
                continue

            sug_category_str = category_map.get(topic.category, "architecture")
            # Validate the category is one of the accepted literals
            _valid_categories = (
                "performance", "quality", "ux",
                "architecture", "testing", "documentation",
            )
            if sug_category_str not in _valid_categories:
                sug_category_str = "architecture"
            sug_category: Literal[
                "performance", "quality", "ux",
                "architecture", "testing", "documentation",
            ] = sug_category_str  # type: ignore[assignment]
            # Map research relevance to suggestion priority
            priority_map: dict[str, Literal["critical", "high", "medium", "low"]] = {
                "high": "high",
                "medium": "medium",
                "low": "low",
            }
            priority = priority_map.get(f.relevance, "medium")

            self._suggestion_reporter.suggest(
                title=f.title,
                description=f"{f.summary}\n\nSource: {f.source}\n\n"
                f"Suggested change: {f.suggested_change}",
                category=sug_category,
                priority=priority,
                affected_module=topic.category,
                evidence=f.source,
            )

    # -- research session ---------------------------------------------------

    def generate_research_prompt(self, topic: ResearchTopic) -> str:
        """Generate a research prompt for a Claude Code agent.

        The returned string is a structured prompt that can be given to
        a sub-agent to perform web searches and code analysis.
        """
        return (
            f"## Research Task: {topic.title}\n\n"
            f"**Category:** {topic.category}\n"
            f"**Priority:** {topic.priority}\n\n"
            f"### Description\n{topic.description}\n\n"
            "### Instructions\n"
            "1. Search for the latest information on this topic\n"
            "2. Check official documentation and release notes\n"
            "3. Look for GitHub issues, discussions, and PRs\n"
            "4. Compare current pipeline implementation against findings\n"
            "5. Identify actionable improvements with evidence\n\n"
            "### Output Format\n"
            "For each finding, provide:\n"
            "- **Title**: Brief descriptive title\n"
            "- **Summary**: 2-3 sentence summary\n"
            "- **Source**: URL or reference\n"
            "- **Relevance**: high/medium/low\n"
            "- **Actionable**: yes/no\n"
            "- **Suggested change**: What to change in the pipeline\n"
        )

    def get_research_summary(self) -> dict[str, object]:
        """Return a summary of the research knowledge base."""
        agenda = self.load_agenda()
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        total_findings = 0
        actionable_findings = 0

        for t in agenda.topics:
            by_status[t.status] = by_status.get(t.status, 0) + 1
            by_category[t.category] = by_category.get(t.category, 0) + 1
            total_findings += len(t.findings)
            actionable_findings += sum(1 for f in t.findings if f.actionable)

        return {
            "total_topics": len(agenda.topics),
            "by_status": by_status,
            "by_category": by_category,
            "total_findings": total_findings,
            "actionable_findings": actionable_findings,
            "last_updated": agenda.last_updated,
        }
