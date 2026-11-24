"""Frozen solver client, robust tool-call parser, and OpenModelica interactive environment."""
from .client import SolverClient, MockClient, Usage
from .parser import parse_tool_call
from .env import TaskEnv, SYSTEM, DIAG_LABELS
from .agent import run_episode, max_steps_for

__all__ = [
    "SolverClient",
    "MockClient",
    "Usage",
    "parse_tool_call",
    "TaskEnv",
    "SYSTEM",
    "DIAG_LABELS",
    "run_episode",
    "max_steps_for",
]
