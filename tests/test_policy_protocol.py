"""CPU checks for the policy boundary; these do not run simulator episodes."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.starvla_astribot.deploy_policy import (
    StarVLAOFTClient,
    adapt_model_action_to_robotwin,
    eval as eval_chunk,
)


class PolicyProtocolTest(unittest.TestCase):
    def setUp(self):
        self.client = StarVLAOFTClient.__new__(StarVLAOFTClient)
        self.client.action_steps = 16
        self.client.clip_grippers = True

    def extract(self, actions):
        return self.client._extract_actions({"ok": True, "data": {"actions": actions}})

    def test_empty_or_invalid_actions_fail_before_rollout(self):
        for actions in (
            np.empty((0, 18)),
            np.zeros((16, 17)),
            np.zeros((0, 16, 18)),
            np.zeros((2, 16, 18)),
            np.full((16, 18), np.nan),
            np.full((16, 18), np.inf),
        ):
            with self.subTest(shape=actions.shape), self.assertRaises(ValueError):
                self.extract(actions)

    def test_valid_chunk_is_clipped_and_limited(self):
        actions = np.full((1, 20, 18), 2.0)
        result = self.extract(actions)
        self.assertEqual(result.shape, (16, 18))
        np.testing.assert_array_equal(result[:, [7, 15]], 1.0)
        np.testing.assert_array_equal(result[:, 16:], 2.0)

    def test_absolute_torso_and_head_targets_become_deltas(self):
        env = SimpleNamespace(
            _get_head_joint_state_now=lambda: [0.0, 0.2],
            _get_torso_joint_state_now=lambda: [0.3],
        )
        action = np.arange(18, dtype=float)
        result = adapt_model_action_to_robotwin(action, env)
        np.testing.assert_array_equal(result[:16], action[:16])
        np.testing.assert_allclose(result[16:], [0.0, 16.8, 15.7])

    def test_chunk_stops_at_success_or_step_limit(self):
        for stop_after, step_limit in ((2, 10), (20, 3)):
            with self.subTest(stop_after=stop_after, step_limit=step_limit):
                env = SimpleNamespace(
                    take_action_cnt=0,
                    step_lim=step_limit,
                    eval_done=False,
                    eval_success=False,
                    get_obs=lambda: {},
                )
                calls = []

                def take_action(action, action_type):
                    calls.append(action)
                    env.take_action_cnt += 1
                    env.eval_success = env.take_action_cnt >= stop_after
                    env.eval_done = env.eval_success

                env.take_action = take_action
                model = SimpleNamespace(
                    get_actions=lambda *_: np.zeros((16, 19)),
                    action_type="qpos",
                    observe_frame=lambda *_: None,
                    log_chunk_timing=lambda **_: None,
                )
                eval_chunk(env, model, {})
                self.assertEqual(len(calls), min(stop_after, step_limit))


if __name__ == "__main__":
    unittest.main()
