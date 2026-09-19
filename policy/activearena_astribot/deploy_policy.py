"""ActiveArena policy adapter entry point.

The frozen implementation keeps its historical module path for checkpoint
compatibility; this module is the public import path for new launchers.
"""
from policy.starvla_astribot.deploy_policy import *  # noqa: F401,F403
