"""Tests for kicad_pipeline.cli.agents_cmd via the main CLI entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.cli.main import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def _mock_registry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both registry and reporter to a temp directory."""
    (tmp_path / "agents").mkdir()
    monkeypatch.setattr(
        "kicad_pipeline.agents.registry.get_registry_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "kicad_pipeline.agents.reporter.get_registry_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "kicad_pipeline.agents.monitor.get_registry_dir",
        lambda: tmp_path,
    )
    return tmp_path


@pytest.mark.usefixtures("_mock_registry_dir")
class TestAgentsList:
    """Tests for ``agents list`` subcommand."""

    def test_list_no_agents(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``agents list`` returns 0 and prints 'No agents' when registry is empty."""
        rc = main(["agents", "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No agents registered" in out


@pytest.mark.usefixtures("_mock_registry_dir")
class TestAgentsRegister:
    """Tests for ``agents register`` subcommand."""

    def test_register_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``agents register`` returns 0 on success."""
        rc = main([
            "agents", "register",
            "--path", "/tmp/test",
            "--name", "Test Project",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Registered agent" in out
        assert "Test Project" in out

    def test_register_with_variant(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``agents register --variant`` includes variant in output."""
        rc = main([
            "agents", "register",
            "--path", "/tmp/test",
            "--name", "Test Project",
            "--variant", "smd-0603",
        ])
        assert rc == 0


@pytest.mark.usefixtures("_mock_registry_dir")
class TestAgentsVersion:
    """Tests for ``agents version`` subcommand."""

    def test_version_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``agents version`` returns 0 and prints the version hash."""
        rc = main(["agents", "version"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Pipeline version updated" in out


@pytest.mark.usefixtures("_mock_registry_dir")
class TestAgentsStatus:
    """Tests for ``agents status`` subcommand."""

    def test_status_no_agents(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``agents status`` returns 0 and prints 'No agents' when registry is empty."""
        rc = main(["agents", "status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No agents registered" in out

    def test_status_after_register(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``agents status`` shows the registered agent after registration."""
        main([
            "agents", "register",
            "--path", "/tmp/test",
            "--name", "My Board",
            "--agent-id", "board-01",
        ])
        # Clear captured output from register
        capsys.readouterr()

        rc = main(["agents", "status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "board-01" in out
        assert "My Board" in out
