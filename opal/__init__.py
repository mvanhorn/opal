"""OPAL (Optimizer with an Agent in the Loop): a coding agent inside your training run, rewriting the optimizer while
it trains."""

from .wrapper import AgentOptimizer, final_program, replay, wrap

__all__ = ["wrap", "replay", "final_program", "AgentOptimizer"]
__version__ = "0.3.3"
