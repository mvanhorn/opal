"""OPAL (OPtimizer with an Agent in the Loop): a coding agent inside your training run, rewriting the optimizer while
it trains."""

from ._version import __version__
from .wrapper import AgentOptimizer, final_program, replay, wrap

__all__ = ["wrap", "replay", "final_program", "AgentOptimizer"]
