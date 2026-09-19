"""Public ActiveArena policy adapter.

The implementation lives in the frozen compatibility module
``policy.starvla_astribot`` so older checkpoints and imports continue to work.
"""
from policy.starvla_astribot.deploy_policy import eval, get_model, reset_model

__all__ = ["eval", "get_model", "reset_model"]
