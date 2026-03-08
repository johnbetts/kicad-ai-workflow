"""CLI commands for the ``agents`` subcommand group.

Provides pipeline-side management of the multi-agent coordination system:
registration, status viewing, bug tracking, and command issuance.
"""

from __future__ import annotations

import argparse  # noqa: TC003 — used at runtime for parser construction
import sys
from pathlib import Path

from kicad_pipeline.agents.models import BugStatus
from kicad_pipeline.exceptions import AgentError


def add_agents_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``agents`` command group with all sub-subcommands."""
    agents_p = subparsers.add_parser("agents", help="Multi-agent coordination")
    agents_sub = agents_p.add_subparsers(dest="agents_command")

    # -- register ----------------------------------------------------------
    reg_p = agents_sub.add_parser("register", help="Register a board agent")
    reg_p.add_argument("--path", required=True, help="Path to board project")
    reg_p.add_argument("--name", required=True, help="Project name")
    reg_p.add_argument("--variant", default=None, help="Active variant")
    reg_p.add_argument("--description", "-d", default="", help="Description")
    reg_p.add_argument("--agent-id", default=None, help="Agent ID (auto-generated if omitted)")

    # -- unregister --------------------------------------------------------
    unreg_p = agents_sub.add_parser("unregister", help="Remove a board agent")
    unreg_p.add_argument("agent_id", help="Agent ID to remove")

    # -- list --------------------------------------------------------------
    agents_sub.add_parser("list", help="List all registered agents")

    # -- status ------------------------------------------------------------
    status_p = agents_sub.add_parser("status", help="Show agent status")
    status_p.add_argument("agent_id", nargs="?", default=None, help="Agent ID (all if omitted)")

    # -- bugs --------------------------------------------------------------
    bugs_p = agents_sub.add_parser("bugs", help="List bugs from agents")
    bugs_p.add_argument("--open-only", action="store_true", help="Show only open bugs")

    # -- rerun -------------------------------------------------------------
    rerun_p = agents_sub.add_parser("rerun", help="Issue rerun command to agent")
    rerun_p.add_argument("agent_id", help="Target agent ID")
    rerun_p.add_argument("--stage", default="pcb", help="Stage to rerun")
    rerun_p.add_argument("--reason", required=True, help="Reason for rerun")

    # -- fix-bug -----------------------------------------------------------
    fix_p = agents_sub.add_parser("fix-bug", help="Mark a bug as fixed")
    fix_p.add_argument("agent_id", help="Agent ID that reported the bug")
    fix_p.add_argument("bug_id", help="Bug ID to mark fixed")
    fix_p.add_argument("--commit", required=True, help="Fix commit hash")

    # -- version -----------------------------------------------------------
    agents_sub.add_parser("version", help="Update pipeline version marker")


def dispatch_agents(args: argparse.Namespace) -> int:
    """Dispatch ``agents`` subcommands. Returns exit code."""
    cmd = getattr(args, "agents_command", None)
    if cmd is None:
        print(
            "Usage: kicad-pipeline agents "
            "{register,unregister,list,status,bugs,rerun,fix-bug,version}"
        )
        return 0

    try:
        if cmd == "register":
            return _cmd_register(args)
        if cmd == "unregister":
            return _cmd_unregister(args)
        if cmd == "list":
            return _cmd_list(args)
        if cmd == "status":
            return _cmd_status(args)
        if cmd == "bugs":
            return _cmd_bugs(args)
        if cmd == "rerun":
            return _cmd_rerun(args)
        if cmd == "fix-bug":
            return _cmd_fix_bug(args)
        if cmd == "version":
            return _cmd_version(args)
    except AgentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Unknown agents command: {cmd}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_register(args: argparse.Namespace) -> int:
    """Handle ``agents register``."""
    import uuid

    from kicad_pipeline.agents.reporter import AgentReporter

    agent_id = args.agent_id or str(uuid.uuid4())[:8]
    reporter = AgentReporter(agent_id)
    reporter.register(
        project_path=str(Path(args.path).resolve()),
        project_name=args.name,
        variant=args.variant,
        description=args.description,
    )
    print(f"Registered agent '{agent_id}' for project '{args.name}'")
    print(f"  Path: {Path(args.path).resolve()}")
    return 0


def _cmd_unregister(args: argparse.Namespace) -> int:
    """Handle ``agents unregister``."""
    import shutil
    from dataclasses import replace

    from kicad_pipeline.agents.registry import get_registry_dir, load_registry, save_registry

    reg_dir = get_registry_dir()
    registry = load_registry(reg_dir / "registry.json")

    found = any(a.agent_id == args.agent_id for a in registry.agents)
    if not found:
        print(f"ERROR: Agent {args.agent_id!r} not found", file=sys.stderr)
        return 1

    new_agents = tuple(a for a in registry.agents if a.agent_id != args.agent_id)
    from datetime import datetime, timezone

    registry = replace(
        registry,
        agents=new_agents,
        updated_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    save_registry(registry, reg_dir / "registry.json")

    # Remove agent directory
    agent_dir = reg_dir / "agents" / args.agent_id
    if agent_dir.exists():
        shutil.rmtree(agent_dir)

    print(f"Unregistered agent '{args.agent_id}'")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle ``agents list``."""
    from kicad_pipeline.agents.registry import get_registry_dir, load_registry

    reg_dir = get_registry_dir()
    registry = load_registry(reg_dir / "registry.json")

    if not registry.agents:
        print("No agents registered.")
        return 0

    for agent in registry.agents:
        variant = f" [{agent.active_variant}]" if agent.active_variant else ""
        print(f"  {agent.agent_id}: {agent.project_name} ({agent.state.value}){variant}")
        print(f"    Path: {agent.project_path}")
        print(f"    Last seen: {agent.last_seen}")

    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle ``agents status``."""
    from kicad_pipeline.agents.monitor import AgentMonitor

    monitor = AgentMonitor()
    results = monitor.scan_all()

    if not results:
        print("No agents registered.")
        return 0

    for reg, status in results:
        if args.agent_id and reg.agent_id != args.agent_id:
            continue

        print(f"Agent: {reg.agent_id} ({reg.project_name})")
        print(f"  State: {reg.state.value}")
        print(f"  Path: {reg.project_path}")

        if status is None:
            print("  Status: <no status file>")
        else:
            print(f"  Updated: {status.updated_at}")
            print(f"  Pipeline version: {status.pipeline_version[:8] or 'unknown'}")
            if status.message:
                print(f"  Message: {status.message}")
            if status.current_stage:
                print(f"  Stage: {status.current_stage}")

            open_bugs = [b for b in status.bugs if b.status == BugStatus.OPEN]
            if open_bugs:
                print(f"  Open bugs: {len(open_bugs)}")
                for bug in open_bugs:
                    print(f"    [{bug.severity.value}] {bug.bug_id}: {bug.title}")

            if status.runs:
                last = status.runs[-1]
                print(f"  Last run: {last.outcome.value} ({last.run_id})")
                if last.drc_summary:
                    drc = last.drc_summary
                    print(
                        f"    DRC: {drc.errors} errors, "
                        f"{drc.warnings} warnings, "
                        f"{drc.unconnected} unconnected"
                    )
        print()

    return 0


def _cmd_bugs(args: argparse.Namespace) -> int:
    """Handle ``agents bugs``."""
    from kicad_pipeline.agents.monitor import AgentMonitor

    monitor = AgentMonitor()

    if args.open_only:
        bugs = monitor.find_open_bugs()
        if not bugs:
            print("No open bugs.")
            return 0
        for agent_id, bug in bugs:
            print(f"  [{bug.severity.value}] {bug.bug_id}: {bug.title}")
            print(f"    Agent: {agent_id}")
            print(f"    Module: {bug.pipeline_module}")
            if bug.description:
                print(f"    Description: {bug.description[:100]}")
            print()
    else:
        results = monitor.scan_all()
        any_bugs = False
        for reg, status in results:
            if status is None or not status.bugs:
                continue
            any_bugs = True
            print(f"Agent: {reg.agent_id} ({reg.project_name})")
            for bug in status.bugs:
                marker = "x" if bug.status in (BugStatus.FIXED, BugStatus.WONT_FIX) else " "
                print(f"  [{marker}] [{bug.severity.value}] {bug.bug_id}: {bug.title}")
                print(f"      Status: {bug.status.value}")
                if bug.fix_commit:
                    print(f"      Fix: {bug.fix_commit}")
            print()
        if not any_bugs:
            print("No bugs reported.")

    return 0


def _cmd_rerun(args: argparse.Namespace) -> int:
    """Handle ``agents rerun``."""
    from kicad_pipeline.agents.monitor import AgentMonitor

    monitor = AgentMonitor()
    monitor.issue_rerun(args.agent_id, args.stage, args.reason)
    print(f"Issued rerun command to agent '{args.agent_id}' (stage: {args.stage})")
    print(f"  Reason: {args.reason}")
    return 0


def _cmd_fix_bug(args: argparse.Namespace) -> int:
    """Handle ``agents fix-bug``."""
    from kicad_pipeline.agents.monitor import AgentMonitor

    monitor = AgentMonitor()
    monitor.issue_bug_update(args.agent_id, args.bug_id, BugStatus.FIXED, args.commit)
    print(f"Marked bug {args.bug_id} as fixed (commit: {args.commit})")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    """Handle ``agents version``."""
    from kicad_pipeline.agents.monitor import AgentMonitor

    monitor = AgentMonitor()
    version = monitor.update_pipeline_version()
    print(f"Pipeline version updated: {version.git_hash[:8]}")
    if version.git_tag:
        print(f"  Tag: {version.git_tag}")
    return 0
