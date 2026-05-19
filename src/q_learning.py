import random

from src.environment import Action


class QLearningAgent:

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.rewards_per_episode = []

    def _ensure_state(self, state):
        if state not in self.Q:
            actions = self.env.get_possible_actions(state)
            self.Q[state] = {a: 0.0 for a in actions}

    def choose_action(self, state):
        if self.env.is_terminal(state):
            return None
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(list(self.Q[state].keys()))
        return max(self.Q[state], key=self.Q[state].__getitem__)

    def update(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)
        max_q_next = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
        td_error = reward + self.gamma * max_q_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self, episodes=1000):
        self.rewards_per_episode = []
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            self.rewards_per_episode.append(total_reward)

    def get_policy(self):
        return {
            state: max(actions, key=actions.__getitem__)
            for state, actions in self.Q.items()
            if actions
        }

    def get_action(self, state):
        if self.env.is_terminal(state):
            return None
        self._ensure_state(state)
        return max(self.Q[state], key=self.Q[state].__getitem__)
