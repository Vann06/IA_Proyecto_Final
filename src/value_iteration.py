"""Value Iteration agent para el hospital estocástico.

Implementa la ecuación de Bellman:
    V(s) = max_a Σ P(s'|s,a) [ R(s,a,s') + γ V(s') ]

Requiere el modelo completo del entorno (probabilidades de transición
y recompensas). El modelo se reconstruye a partir de la lógica de
`HospitalEnvironment.step` sin invocar la simulación estocástica.
"""

from src.environment import (
    Action,
    CHARGE_POS,
    CHARGE_AMOUNT,
    GOAL_POS,
    CELL_REWARDS,
    W,
)


class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-3, max_iterations=1000):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = {}
        self.policy = {}
        self.iterations = 0

        self._perp_left = {
            Action.UP:    Action.LEFT,
            Action.DOWN:  Action.RIGHT,
            Action.LEFT:  Action.DOWN,
            Action.RIGHT: Action.UP,
        }
        self._perp_right = {
            Action.UP:    Action.RIGHT,
            Action.DOWN:  Action.LEFT,
            Action.LEFT:  Action.UP,
            Action.RIGHT: Action.DOWN,
        }

    # ── Modelo de transición ───────────────────────────────────
    def _apply_move(self, row, col, action):
        from src.environment import MOVE_DELTAS
        dr, dc = MOVE_DELTAS[action]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < self.env.rows and 0 <= nc < self.env.cols):
            return row, col
        if self.env.map[nr][nc] == W:
            return row, col
        return nr, nc

    def _resolve_outcome(self, state, actual):
        """Aplica la acción 'actual' (ya resuelta la estocasticidad)
        y devuelve (next_state, reward, done)."""
        row, col, battery = state

        if actual == Action.CHARGE:
            new_row, new_col = row, col
            if (row, col) == CHARGE_POS:
                battery = min(battery + CHARGE_AMOUNT, self.env.max_battery)
            # CHARGE fuera de estación: no descuenta batería
        else:
            new_row, new_col = self._apply_move(row, col, actual)
            battery -= 1

        cell_type = self.env.map[new_row][new_col]
        reward = CELL_REWARDS[cell_type]

        done = False
        if (new_row, new_col) == GOAL_POS:
            done = True
        elif battery <= 0:
            battery = 0
            reward = -50
            done = True

        return (new_row, new_col, battery), reward, done

    def _get_transitions(self, state, action):
        """Lista de (prob, next_state, reward, done) para (state, action).

        Replica la estocasticidad de env._actual_action:
          - STAY / CHARGE: deterministas (prob 1.0)
          - UP/DOWN/LEFT/RIGHT: 0.8 deseada, 0.1 perp-izq, 0.1 perp-der
        """
        if action in (Action.STAY, Action.CHARGE):
            outcome = self._resolve_outcome(state, action)
            return [(1.0, *outcome)]

        outcomes = [
            (0.8, action),
            (0.1, self._perp_left[action]),
            (0.1, self._perp_right[action]),
        ]
        transitions = []
        for prob, actual in outcomes:
            next_state, reward, done = self._resolve_outcome(state, actual)
            transitions.append((prob, next_state, reward, done))
        return transitions

    # ── Bellman ────────────────────────────────────────────────
    def compute_q_value(self, state, action):
        """Q(s,a) = Σ P(s'|s,a) [ R + γ V(s') ]"""
        q = 0.0
        for prob, next_state, reward, done in self._get_transitions(state, action):
            v_next = 0.0 if done else self.V.get(next_state, 0.0)
            q += prob * (reward + self.gamma * v_next)
        return q

    def _best_value(self, state):
        """Mejor V(s) tomando el max sobre las acciones posibles.
        Devuelve 0.0 si el estado es terminal o no tiene acciones."""
        if self.env.is_terminal(state):
            return 0.0
        actions = self.env.get_possible_actions(state)
        if not actions:
            return 0.0
        return max(self.compute_q_value(state, a) for a in actions)

    def run(self, verbose=False):
        """Itera la ecuación de Bellman hasta convergencia."""
        states = self.env.get_all_states()
        self.V = dict.fromkeys(states, 0.0)

        for it in range(1, self.max_iterations + 1):
            delta = 0.0
            new_v = {}
            for s in states:
                best = self._best_value(s)
                new_v[s] = best
                delta = max(delta, abs(best - self.V[s]))

            self.V = new_v
            self.iterations = it

            if verbose and it % 10 == 0:
                print(f"Iter {it:4d}  Δ = {delta:.6f}")

            if delta < self.theta:
                if verbose:
                    print(f"Convergió en {it} iteraciones (Δ = {delta:.6f})")
                break

        self.extract_policy()
        return self.V

    def extract_policy(self):
        """Extrae política greedy a partir de V."""
        self.policy = {}
        for s in self.env.get_all_states():
            if self.env.is_terminal(s):
                continue
            actions = self.env.get_possible_actions(s)
            if not actions:
                continue
            best_action = max(actions, key=lambda a, s=s: self.compute_q_value(s, a))
            self.policy[s] = best_action
        return self.policy

    def get_action(self, state):
        """Devuelve la acción recomendada por la política. STAY si no hay."""
        if self.env.is_terminal(state):
            return None
        return self.policy.get(state, Action.STAY)
