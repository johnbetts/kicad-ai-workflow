"""Multi-agent coordination for distributed KiCad pipeline sessions."""

# Expose submodules
from . import executor, models, planner, reporter, status, validator

# Re-export commonly used functions
from .executor.commands import (
    _command_from_dict,
    _command_to_dict,
    acknowledge_command,
    issue_command,
    load_commands,
    save_commands,
)
