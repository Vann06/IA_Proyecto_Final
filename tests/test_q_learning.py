import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.q_learning import QLearningAgent
from src.environment import (
    HospitalEnvironment, Action,
    START_POS, GOAL_POS, CHARGE_POS, MAX_BATTERY,
)


class TestQLearningInit(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_default_alpha(self):
        agent = QLearningAgent(self.env)
        self.assertEqual(agent.alpha, 0.1)

    def test_default_gamma(self):
        agent = QLearningAgent(self.env)
        self.assertEqual(agent.gamma, 0.9)

    def test_default_epsilon(self):
        agent = QLearningAgent(self.env)
        self.assertEqual(agent.epsilon, 0.2)

    def test_custom_params(self):
        agent = QLearningAgent(self.env, alpha=0.5, gamma=0.8, epsilon=0.1)
        self.assertEqual(agent.alpha, 0.5)
        self.assertEqual(agent.gamma, 0.8)
        self.assertEqual(agent.epsilon, 0.1)

    def test_q_table_empty(self):
        agent = QLearningAgent(self.env)
        self.assertEqual(agent.Q, {})

    def test_env_stored(self):
        agent = QLearningAgent(self.env)
        self.assertIs(agent.env, self.env)

    def test_rewards_history_empty(self):
        agent = QLearningAgent(self.env)
        self.assertEqual(agent.rewards_per_episode, [])


class TestChooseAction(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.agent = QLearningAgent(self.env, epsilon=0.2)

    def test_terminal_goal_returns_none(self):
        state = (GOAL_POS[0], GOAL_POS[1], 10)
        self.assertIsNone(self.agent.choose_action(state))

    def test_terminal_battery_zero_returns_none(self):
        state = (1, 1, 0)
        self.assertIsNone(self.agent.choose_action(state))

    def test_exploration_returns_random_action(self):
        # epsilon=0.5 → explora cuando random < 0.5
        agent = QLearningAgent(self.env, epsilon=0.5)
        state = (1, 1, MAX_BATTERY)
        with patch('src.q_learning.random.random', return_value=0.3), \
             patch('src.q_learning.random.choice', return_value=Action.DOWN) as mock_choice:
            action = agent.choose_action(state)
        self.assertEqual(action, Action.DOWN)
        mock_choice.assert_called_once()

    def test_exploitation_returns_best_q_action(self):
        state = (1, 1, MAX_BATTERY)
        self.agent._ensure_state(state)
        self.agent.Q[state][Action.RIGHT] = 5.0
        with patch('src.q_learning.random.random', return_value=0.9):  # >= 0.2 → explota
            action = self.agent.choose_action(state)
        self.assertEqual(action, Action.RIGHT)

    def test_unseen_state_returns_valid_action_no_exception(self):
        state = (1, 1, MAX_BATTERY)
        with patch('src.q_learning.random.random', return_value=0.9):
            action = self.agent.choose_action(state)
        self.assertIn(action, self.env.get_possible_actions(state))

    def test_ensure_state_called_for_new_state(self):
        state = (1, 1, MAX_BATTERY)
        self.assertNotIn(state, self.agent.Q)
        self.agent.choose_action(state)
        self.assertIn(state, self.agent.Q)

    def test_charge_action_included_at_station(self):
        state = (CHARGE_POS[0], CHARGE_POS[1], 10)
        with patch('src.q_learning.random.random', return_value=0.9):
            action = self.agent.choose_action(state)
        valid = self.env.get_possible_actions(state)
        self.assertIn(action, valid)
        self.assertIn(Action.CHARGE, valid)


class TestUpdate(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.agent = QLearningAgent(self.env, alpha=0.1, gamma=0.9)

    def test_basic_update_q_value(self):
        state = (1, 1, MAX_BATTERY)
        next_state = (1, 2, MAX_BATTERY - 1)
        self.agent._ensure_state(state)
        self.agent._ensure_state(next_state)
        # Q[s][a]=0, max_Q[s']=0, reward=-1 → nuevo Q = 0 + 0.1*(-1+0-0) = -0.1
        self.agent.update(state, Action.RIGHT, -1.0, next_state)
        self.assertAlmostEqual(self.agent.Q[state][Action.RIGHT], -0.1)

    def test_bellman_formula_full(self):
        state = (1, 1, 15)
        next_state = (1, 2, 14)
        self.agent._ensure_state(state)
        self.agent._ensure_state(next_state)
        self.agent.Q[state][Action.RIGHT] = 2.0
        self.agent.Q[next_state][Action.UP] = 10.0
        # Q = 2.0 + 0.1*(-1 + 0.9*10 - 2.0) = 2.0 + 0.1*6.0 = 2.6
        self.agent.update(state, Action.RIGHT, -1.0, next_state)
        self.assertAlmostEqual(self.agent.Q[state][Action.RIGHT], 2.6)

    def test_terminal_next_state_no_future_reward(self):
        state = (10, 13, 5)
        next_state = (GOAL_POS[0], GOAL_POS[1], 4)  # terminal
        self.agent._ensure_state(state)
        # max_Q_next = 0.0 porque terminal → Q = 0 + 0.1*(100 + 0 - 0) = 10.0
        self.agent.update(state, Action.RIGHT, 100.0, next_state)
        self.assertAlmostEqual(self.agent.Q[state][Action.RIGHT], 10.0)

    def test_new_states_initialized_no_exception(self):
        state = (3, 3, 20)
        next_state = (3, 4, 19)
        self.assertNotIn(state, self.agent.Q)
        self.assertNotIn(next_state, self.agent.Q)
        self.agent.update(state, Action.RIGHT, -1.0, next_state)
        self.assertIn(state, self.agent.Q)
        self.assertIn(next_state, self.agent.Q)

    def test_only_target_action_updated(self):
        state = (1, 1, MAX_BATTERY)
        next_state = (1, 2, MAX_BATTERY - 1)
        self.agent._ensure_state(state)
        self.agent._ensure_state(next_state)
        self.agent.Q[state][Action.UP] = 5.0
        self.agent.Q[state][Action.DOWN] = 3.0
        self.agent.update(state, Action.RIGHT, -1.0, next_state)
        self.assertAlmostEqual(self.agent.Q[state][Action.UP], 5.0)
        self.assertAlmostEqual(self.agent.Q[state][Action.DOWN], 3.0)


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.agent = QLearningAgent(self.env)

    def test_correct_number_of_episodes(self):
        self.agent.train(episodes=10)
        self.assertEqual(len(self.agent.rewards_per_episode), 10)

    def test_default_episodes_is_1000(self):
        self.agent.train()
        self.assertEqual(len(self.agent.rewards_per_episode), 1000)

    def test_train_returns_none(self):
        result = self.agent.train(episodes=1)
        self.assertIsNone(result)

    def test_q_table_populated_after_train(self):
        self.agent.train(episodes=5)
        self.assertGreater(len(self.agent.Q), 0)

    def test_rewards_are_numeric(self):
        self.agent.train(episodes=3)
        for r in self.agent.rewards_per_episode:
            self.assertIsInstance(r, (int, float))

    def test_train_resets_rewards_history(self):
        self.agent.rewards_per_episode = [1.0, 2.0]
        self.agent.train(episodes=3)
        self.assertEqual(len(self.agent.rewards_per_episode), 3)

    def test_single_episode_terminates(self):
        # verifica que el episodio siempre alcanza done=True (no loop infinito)
        self.agent.train(episodes=1)
        self.assertEqual(len(self.agent.rewards_per_episode), 1)


class TestGetPolicy(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.agent = QLearningAgent(self.env)

    def test_returns_dict(self):
        self.agent.train(episodes=5)
        self.assertIsInstance(self.agent.get_policy(), dict)

    def test_keys_match_q_table_states(self):
        self.agent.train(episodes=5)
        policy = self.agent.get_policy()
        expected_keys = {s for s, actions in self.agent.Q.items() if actions}
        self.assertEqual(set(policy.keys()), expected_keys)

    def test_values_are_actions(self):
        self.agent.train(episodes=5)
        for action in self.agent.get_policy().values():
            self.assertIsInstance(action, Action)

    def test_returns_argmax_action(self):
        state = (1, 1, MAX_BATTERY)
        self.agent._ensure_state(state)
        self.agent.Q[state][Action.UP] = 1.0
        self.agent.Q[state][Action.RIGHT] = 5.0
        self.agent.Q[state][Action.DOWN] = 2.0
        policy = self.agent.get_policy()
        self.assertEqual(policy[state], Action.RIGHT)

    def test_empty_q_table_returns_empty_dict(self):
        self.assertEqual(self.agent.get_policy(), {})


class TestGetAction(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.agent = QLearningAgent(self.env)

    def test_terminal_goal_returns_none(self):
        state = (GOAL_POS[0], GOAL_POS[1], 10)
        self.assertIsNone(self.agent.get_action(state))

    def test_terminal_battery_zero_returns_none(self):
        state = (1, 1, 0)
        self.assertIsNone(self.agent.get_action(state))

    def test_returns_best_q_action(self):
        state = (1, 1, MAX_BATTERY)
        self.agent._ensure_state(state)
        self.agent.Q[state][Action.UP] = 1.0
        self.agent.Q[state][Action.RIGHT] = 8.0
        self.agent.Q[state][Action.DOWN] = 2.0
        self.assertEqual(self.agent.get_action(state), Action.RIGHT)

    def test_unseen_state_returns_valid_action(self):
        state = (7, 7, MAX_BATTERY)
        action = self.agent.get_action(state)
        self.assertIsNotNone(action)
        self.assertIn(action, self.env.get_possible_actions(state))

    def test_interface_compatible_with_value_iteration(self):
        from src.value_iteration import ValueIterationAgent
        vi_agent = ValueIterationAgent(self.env)
        vi_agent.run()
        self.agent.train(episodes=5)
        state = (1, 1, MAX_BATTERY)
        vi_action = vi_agent.get_action(state)
        ql_action = self.agent.get_action(state)
        # ambos deben ser Action o None — no importa que coincidan en valor
        self.assertTrue(vi_action is None or isinstance(vi_action, Action))
        self.assertTrue(ql_action is None or isinstance(ql_action, Action))


if __name__ == '__main__':
    unittest.main(verbosity=2)
