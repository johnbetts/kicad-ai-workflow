"""Verification orchestrator — Python-driven, per-component, dual-persona.

Replaces prompt-only skill files with programmatic orchestration.
The Python script IS the process: it decides what runs, validates output,
tracks known bugs, and enforces per-component granularity.
"""

from __future__ import annotations
