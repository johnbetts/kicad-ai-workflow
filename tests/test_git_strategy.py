"""Tests for kicad_pipeline.orchestrator.git_strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.orchestrator.git_strategy import (
    commit_revision,
    commit_stage,
    tag_release,
    tag_revision,
)
from kicad_pipeline.orchestrator.models import StageId

_GIT_OPS = "kicad_pipeline.orchestrator.git_strategy.git_ops"


class TestCommitStage:
    @patch(f"{_GIT_OPS}.commit", return_value=True)
    @patch(f"{_GIT_OPS}.stage_files", return_value=True)
    def test_calls_git_ops(
        self,
        mock_stage: object,
        mock_commit: object,
        tmp_path: Path,
    ) -> None:
        result = commit_stage(
            tmp_path, "standard-0805", StageId.SCHEMATIC, "generate schematic"
        )
        assert result is True

    @patch(f"{_GIT_OPS}.commit", return_value=True)
    @patch(f"{_GIT_OPS}.stage_files", return_value=True)
    def test_message_format(
        self,
        mock_stage: object,
        mock_commit: object,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock

        mock_commit = MagicMock(return_value=True)
        with patch(f"{_GIT_OPS}.commit", mock_commit), \
             patch(f"{_GIT_OPS}.stage_files", return_value=True):
            commit_stage(
                tmp_path, "standard-0805", StageId.PCB, "layout components"
            )
        mock_commit.assert_called_once()
        message = mock_commit.call_args[0][0]
        assert message == "feat(pcb/standard-0805): layout components"

    @patch(f"{_GIT_OPS}.commit", return_value=True)
    @patch(f"{_GIT_OPS}.stage_files", return_value=True)
    def test_stages_variant_path(
        self,
        mock_stage: object,
        mock_commit: object,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock

        mock_stage_fn = MagicMock(return_value=True)
        with patch(f"{_GIT_OPS}.stage_files", mock_stage_fn), \
             patch(f"{_GIT_OPS}.commit", return_value=True):
            commit_stage(
                tmp_path, "compact-0603", StageId.REQUIREMENTS, "parse spec"
            )
        mock_stage_fn.assert_called_once()
        staged_files = mock_stage_fn.call_args[0][0]
        assert staged_files == ["variants/compact-0603/"]


class TestTagRevision:
    @patch(f"{_GIT_OPS}.create_tag", return_value=True)
    def test_format(self, mock_tag: object, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        mock_tag_fn = MagicMock(return_value=True)
        with patch(f"{_GIT_OPS}.create_tag", mock_tag_fn):
            result = tag_revision(tmp_path, "standard-0805", 3)
        assert result is True
        mock_tag_fn.assert_called_once()
        assert mock_tag_fn.call_args[0][0] == "standard-0805/rev3"


class TestTagRelease:
    @patch(f"{_GIT_OPS}.create_tag", return_value=True)
    def test_format(self, mock_tag: object, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        mock_tag_fn = MagicMock(return_value=True)
        with patch(f"{_GIT_OPS}.create_tag", mock_tag_fn):
            result = tag_release(tmp_path, "standard-0805", "v1.0.0")
        assert result is True
        mock_tag_fn.assert_called_once()
        assert mock_tag_fn.call_args[0][0] == "standard-0805/v1.0.0"


class TestCommitRevision:
    @patch(f"{_GIT_OPS}.commit", return_value=True)
    @patch(f"{_GIT_OPS}.stage_files", return_value=True)
    def test_message_format(
        self,
        mock_stage: object,
        mock_commit: object,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock

        mock_commit_fn = MagicMock(return_value=True)
        with patch(f"{_GIT_OPS}.commit", mock_commit_fn), \
             patch(f"{_GIT_OPS}.stage_files", return_value=True):
            commit_revision(tmp_path, "standard-0805", 2)
        mock_commit_fn.assert_called_once()
        message = mock_commit_fn.call_args[0][0]
        assert message == "feat(production/standard-0805): revision 2"
