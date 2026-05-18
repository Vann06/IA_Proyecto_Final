import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment import (
    HospitalEnvironment, Action,
    START_POS, GOAL_POS, CHARGE_POS,
    MAX_BATTERY, CHARGE_AMOUNT, HOSPITAL_MAP, W,
)


class TestReset(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_starts_at_correct_position(self):
        state = self.env.reset()
        row, col, _ = state
        self.assertEqual((row, col), START_POS)

    def test_starts_with_full_battery(self):
        state = self.env.reset()
        _, _, battery = state
        self.assertEqual(battery, MAX_BATTERY)

    def test_reset_after_steps_restores_state(self):
        self.env.reset()
        with patch('random.random', return_value=0.0):
            self.env.step(Action.DOWN)
            self.env.step(Action.DOWN)
        state = self.env.reset()
        row, col, battery = state
        self.assertEqual((row, col), START_POS)
        self.assertEqual(battery, MAX_BATTERY)


class TestWallCollision(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.env.reset()  # estado inicial: (1, 1, MAX_BATTERY)

    def test_robot_blocked_by_wall_above(self):
        # (1,1) → UP → (0,1) es W, debe quedarse en (1,1)
        with patch('random.random', return_value=0.0):
            state, _, _, _ = self.env.step(Action.UP)
        row, col, _ = state
        self.assertEqual((row, col), START_POS)

    def test_robot_blocked_by_wall_left(self):
        # (1,1) → LEFT → (1,0) es W, debe quedarse en (1,1)
        with patch('random.random', return_value=0.0):
            state, _, _, _ = self.env.step(Action.LEFT)
        row, col, _ = state
        self.assertEqual((row, col), START_POS)

    def test_battery_decreases_even_when_blocked(self):
        # Golpear una pared igual cuesta batería
        with patch('random.random', return_value=0.0):
            state, _, _, _ = self.env.step(Action.UP)
        _, _, battery = state
        self.assertEqual(battery, MAX_BATTERY - 1)


class TestBatteryDrain(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()
        self.env.reset()

    def test_battery_decreases_on_movement(self):
        with patch('random.random', return_value=0.0):
            state, _, _, _ = self.env.step(Action.DOWN)
        _, _, battery = state
        self.assertEqual(battery, MAX_BATTERY - 1)

    def test_battery_decreases_on_stay(self):
        state, _, _, _ = self.env.step(Action.STAY)
        _, _, battery = state
        self.assertEqual(battery, MAX_BATTERY - 1)

    def test_battery_zero_triggers_done(self):
        self.env.state = (1, 1, 1)
        state, reward, done, _ = self.env.step(Action.STAY)
        _, _, battery = state
        self.assertEqual(battery, 0)
        self.assertTrue(done)
        self.assertEqual(reward, -50)


class TestCharging(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_charge_increases_battery_at_station(self):
        self.env.state = (CHARGE_POS[0], CHARGE_POS[1], 5)
        state, _, _, _ = self.env.step(Action.CHARGE)
        _, _, battery = state
        self.assertEqual(battery, 5 + CHARGE_AMOUNT)

    def test_charge_does_not_exceed_max_battery(self):
        self.env.state = (CHARGE_POS[0], CHARGE_POS[1], MAX_BATTERY - 2)
        state, _, _, _ = self.env.step(Action.CHARGE)
        _, _, battery = state
        self.assertEqual(battery, MAX_BATTERY)

    def test_charge_outside_station_does_nothing(self):
        self.env.state = (1, 1, 10)
        state, _, _, _ = self.env.step(Action.CHARGE)
        _, _, battery = state
        self.assertEqual(battery, 10)

    def test_charge_does_not_move_robot(self):
        self.env.state = (CHARGE_POS[0], CHARGE_POS[1], 10)
        state, _, _, _ = self.env.step(Action.CHARGE)
        row, col, _ = state
        self.assertEqual((row, col), CHARGE_POS)


class TestTerminalConditions(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_reaching_goal_ends_episode(self):
        # (10,13) es pasillo, moverse RIGHT llega a GOAL (10,14)
        self.env.state = (10, 13, 10)
        with patch('random.random', return_value=0.0):
            state, reward, done, _ = self.env.step(Action.RIGHT)
        row, col, _ = state
        self.assertEqual((row, col), GOAL_POS)
        self.assertTrue(done)
        self.assertEqual(reward, 100)

    def test_battery_zero_ends_episode_with_penalty(self):
        self.env.state = (1, 1, 1)
        _, reward, done, _ = self.env.step(Action.STAY)
        self.assertTrue(done)
        self.assertEqual(reward, -50)

    def test_is_terminal_at_goal(self):
        self.assertTrue(self.env.is_terminal((GOAL_POS[0], GOAL_POS[1], 10)))

    def test_is_terminal_battery_zero(self):
        self.assertTrue(self.env.is_terminal((1, 1, 0)))

    def test_is_not_terminal_normal_state(self):
        self.assertFalse(self.env.is_terminal((1, 1, 10)))


class TestRewards(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_pasillo_reward(self):
        # (1,1) → RIGHT → (1,2) es pasillo P → reward -1
        self.env.state = (1, 1, 10)
        with patch('random.random', return_value=0.0):
            _, reward, _, _ = self.env.step(Action.RIGHT)
        self.assertEqual(reward, -1)

    def test_zona_congestionada_reward(self):
        # (1,3) → RIGHT → (1,4) es zona Z → reward -5
        self.env.state = (1, 3, 10)
        with patch('random.random', return_value=0.0):
            _, reward, _, _ = self.env.step(Action.RIGHT)
        self.assertEqual(reward, -5)

    def test_zona_riesgo_reward(self):
        # (3,6) → RIGHT → (3,7) es zona R → reward -20
        self.env.state = (3, 6, 10)
        with patch('random.random', return_value=0.0):
            _, reward, _, _ = self.env.step(Action.RIGHT)
        self.assertEqual(reward, -20)


class TestGetAllStates(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_no_wall_cells_in_states(self):
        for row, col, _ in self.env.get_all_states():
            self.assertNotEqual(HOSPITAL_MAP[row][col], W)

    def test_battery_range_correct(self):
        batteries = [b for _, _, b in self.env.get_all_states()]
        self.assertEqual(min(batteries), 0)
        self.assertEqual(max(batteries), MAX_BATTERY)

    def test_total_count_correct(self):
        non_wall = sum(
            1 for r in range(len(HOSPITAL_MAP))
            for c in range(len(HOSPITAL_MAP[0]))
            if HOSPITAL_MAP[r][c] != W
        )
        self.assertEqual(len(self.env.get_all_states()), non_wall * (MAX_BATTERY + 1))


class TestGetPossibleActions(unittest.TestCase):

    def setUp(self):
        self.env = HospitalEnvironment()

    def test_terminal_goal_returns_empty(self):
        self.assertEqual(self.env.get_possible_actions((GOAL_POS[0], GOAL_POS[1], 10)), [])

    def test_terminal_battery_returns_empty(self):
        self.assertEqual(self.env.get_possible_actions((1, 1, 0)), [])

    def test_charge_available_at_station(self):
        actions = self.env.get_possible_actions((CHARGE_POS[0], CHARGE_POS[1], 10))
        self.assertIn(Action.CHARGE, actions)

    def test_charge_not_available_outside_station(self):
        actions = self.env.get_possible_actions((1, 1, 10))
        self.assertNotIn(Action.CHARGE, actions)

    def test_normal_state_has_five_actions(self):
        actions = self.env.get_possible_actions((1, 1, 10))
        self.assertEqual(len(actions), 5)

    def test_charge_state_has_six_actions(self):
        actions = self.env.get_possible_actions((CHARGE_POS[0], CHARGE_POS[1], 10))
        self.assertEqual(len(actions), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
