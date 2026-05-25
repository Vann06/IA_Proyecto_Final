import random

from src.environment import HospitalEnvironment
from src.value_iteration import ValueIterationAgent
from src.q_learning import QLearningAgent
from src.metrics import evaluate_agent


def train_value_iteration(env, gamma=0.9, theta=1e-3, max_iterations=1000, verbose=False):
    """Entrena y devuelve un ValueIterationAgent."""
    agent = ValueIterationAgent(env, gamma=gamma, theta=theta, max_iterations=max_iterations)
    agent.run(verbose=verbose)
    return agent


def train_q_learning(env, episodes=50_000, alpha=0.1, gamma=0.95, epsilon=0.3,
                     max_steps=200, seed=42):
    """Entrena y devuelve un QLearningAgent."""
    random.seed(seed)
    agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon)
    agent.train(episodes=episodes, max_steps=max_steps)
    return agent


def train_q_learning_with_decay(env, episodes=50_000, alpha=0.1, gamma=0.95,
                                epsilon_start=0.7, epsilon_end=0.05,
                                max_steps=200, seed=42):
    """Entrena Q-Learning con reducción gradual de epsilon."""
    random.seed(seed)
    agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon_start)

    agent.rewards_per_episode = []
    agent.steps_per_episode = []

    for episode in range(episodes):
        progress = episode / (episodes - 1) if episodes > 1 else 1.0
        agent.epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)

        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

        agent.rewards_per_episode.append(total_reward)
        agent.steps_per_episode.append(steps)

    agent.epsilon = epsilon_end
    return agent


def evaluate_model(env, agent, name, episodes=100, max_steps=200, seed=123):
    """Evalúa un agente y devuelve el diccionario de resultados de metrics.py."""
    random.seed(seed)
    return evaluate_agent(env, agent, episodes=episodes, max_steps=max_steps, agent_name=name)


Q_LEARNING_CONFIGS = [
    {
        "name": "Q-Learning base",
        "type": "standard",
        "episodes": 50_000,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.30,
        "seed": 42,
    },
    {
        "name": "Q-Learning más episodios",
        "type": "standard",
        "episodes": 80_000,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.30,
        "seed": 44,
    },
    {
        "name": "Q-Learning gamma alto",
        "type": "standard",
        "episodes": 80_000,
        "alpha": 0.1,
        "gamma": 0.98,
        "epsilon": 0.30,
        "seed": 45,
    },
    {
        "name": "Q-Learning más explotación",
        "type": "standard",
        "episodes": 80_000,
        "alpha": 0.1,
        "gamma": 0.98,
        "epsilon": 0.15,
        "seed": 46,
    },
    {
        "name": "Q-Learning epsilon decay",
        "type": "decay",
        "episodes": 80_000,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon_start": 0.7,
        "epsilon_end": 0.05,
        "seed": 47,
    },
]


def run_q_experiment(config, env, max_steps=200):
    """Entrena un agente Q-Learning según la configuración dada."""
    if config["type"] == "decay":
        return train_q_learning_with_decay(
            env,
            episodes=config["episodes"],
            alpha=config["alpha"],
            gamma=config["gamma"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            max_steps=max_steps,
            seed=config["seed"],
        )
    return train_q_learning(
        env,
        episodes=config["episodes"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        max_steps=max_steps,
        seed=config["seed"],
    )


def pick_best_q_config(results_list, experiment_configs):
    """
    Recibe la lista de resultados de evaluate_model y los configs correspondientes.
    Devuelve el config con mayor éxito, luego mayor recompensa, luego menos fallos.
    """
    ranked = sorted(
        zip(results_list, experiment_configs),
        key=lambda x: (
            x[0]["success_rate"],
            x[0]["avg_reward"],
            -x[0]["mission_failures"],
        ),
        reverse=True,
    )
    return ranked[0][1]
