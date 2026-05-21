"""
Módulo de métricas para comparar agentes en el entorno HospitalEnvironment.

Este archivo no implementa Value Iteration ni Q-Learning.
Su responsabilidad es evaluar agentes ya entrenados o ya calculados
bajo las mismas condiciones del entorno.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _get_action_from_agent(agent: Any, state: Any) -> Any:
    """
    Obtiene la acción recomendada por un agente.

    Se intenta usar get_action primero porque normalmente representa
    la política final del agente. Si no existe, se usa choose_action.
    """
    if hasattr(agent, "get_action"):
        return agent.get_action(state)

    if hasattr(agent, "act"):
        return agent.act(state)

    if hasattr(agent, "choose_action"):
        return agent.choose_action(state)

    raise AttributeError(
        "El agente debe tener un método get_action(state), act(state) "
        "o choose_action(state)."
    )


def _extract_battery(state: Any, env: Any = None) -> Optional[float]:
    """
    Intenta obtener el nivel de batería desde el estado.

    Según el README, el estado esperado es:
        (row, col, battery)

    Si el entorno cambia, también se intenta leer desde atributos del env.
    """
    if isinstance(state, tuple) and len(state) >= 3:
        battery = state[2]
        if isinstance(battery, (int, float)):
            return float(battery)

    if env is not None:
        for attr in ["battery", "current_battery", "battery_level"]:
            if hasattr(env, attr):
                value = getattr(env, attr)
                if isinstance(value, (int, float)):
                    return float(value)

    return None


def _parse_step_output(step_output: Any):
    """
    Soporta dos formatos comunes:

    Formato clásico:
        next_state, reward, done, info

    Formato Gymnasium:
        next_state, reward, terminated, truncated, info
    """
    if not isinstance(step_output, tuple):
        raise ValueError("env.step(action) debe retornar una tupla.")

    if len(step_output) == 4:
        next_state, reward, done, info = step_output
        return next_state, reward, done, info

    if len(step_output) == 5:
        next_state, reward, terminated, truncated, info = step_output
        done = terminated or truncated
        return next_state, reward, done, info

    raise ValueError(
        "env.step(action) debe retornar 4 valores "
        "(next_state, reward, done, info) o 5 valores."
    )


def _is_success(info: Dict[str, Any], total_reward: float, done: bool) -> bool:
    """
    Determina si la misión fue exitosa.

    Lo ideal es que environment.py devuelva algo como:
        info = {"success": True}

    Si no existe esa llave, se usa una regla aproximada:
    episodio terminado con recompensa total positiva.
    """
    if isinstance(info, dict):
        for key in ["success", "delivered", "goal_reached", "delivery_success"]:
            if key in info:
                return bool(info[key])

        reason = info.get("terminal_reason") or info.get("done_reason")
        if reason in ["goal", "success", "delivery", "delivered"]:
            return True

    return bool(done and total_reward > 0)


def evaluate_agent(
    env: Any,
    agent: Any,
    episodes: int = 100,
    max_steps: int = 500,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evalúa un agente en varios episodios.

    Métricas calculadas:
    - tiempo promedio de entrega
    - consumo promedio de batería
    - porcentaje de entregas exitosas
    - recompensa acumulada promedio
    - cantidad de fallos de misión
    """

    if agent_name is None:
        agent_name = agent.__class__.__name__

    # Para evaluar Q-Learning de forma justa, se intenta apagar exploración.
    old_epsilon = getattr(agent, "epsilon", None)
    if old_epsilon is not None:
        agent.epsilon = 0.0

    episode_results: List[Dict[str, Any]] = []

    for episode in range(episodes):
        state = env.reset()
        initial_battery = _extract_battery(state, env)

        total_reward = 0.0
        steps = 0
        done = False
        last_info: Dict[str, Any] = {}

        while not done and steps < max_steps:
            action = _get_action_from_agent(agent, state)

            if action is None:
                break

            step_output = env.step(action)
            next_state, reward, done, info = _parse_step_output(step_output)

            total_reward += float(reward)
            steps += 1
            state = next_state
            last_info = info if isinstance(info, dict) else {}

        final_battery = _extract_battery(state, env)

        if initial_battery is not None and final_battery is not None:
            battery_consumed = max(0.0, initial_battery - final_battery)
        else:
            battery_consumed = None

        success = _is_success(last_info, total_reward, done)

        episode_results.append(
            {
                "agent": agent_name,
                "episode": episode + 1,
                "steps": steps,
                "total_reward": total_reward,
                "battery_consumed": battery_consumed,
                "success": success,
                "failure": not success,
            }
        )

    # Restaurar epsilon original si existía.
    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    details = pd.DataFrame(episode_results)

    success_rate = details["success"].mean() * 100
    failures = int(details["failure"].sum())

    successful_deliveries = details[details["success"] == True]

    if len(successful_deliveries) > 0:
        avg_delivery_time = successful_deliveries["steps"].mean()
    else:
        avg_delivery_time = None

    avg_battery = details["battery_consumed"].dropna().mean()
    avg_reward = details["total_reward"].mean()

    return {
        "agent": agent_name,
        "episodes": episodes,
        "avg_delivery_time": avg_delivery_time,
        "avg_battery_consumption": avg_battery,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "mission_failures": failures,
        "details": details,
    }


def compare_agents(*results: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea una tabla comparativa entre varios agentes.
    """

    rows = []

    for result in results:
        rows.append(
            {
                "Agente": result["agent"],
                "Episodios": result["episodes"],
                "Tiempo promedio de entrega": result["avg_delivery_time"],
                "Consumo promedio de batería": result["avg_battery_consumption"],
                "% entregas exitosas": result["success_rate"],
                "Recompensa acumulada promedio": result["avg_reward"],
                "Fallos de misión": result["mission_failures"],
            }
        )

    return pd.DataFrame(rows)


def save_results_table(
    comparison_df: pd.DataFrame,
    path: str | Path = "results/tables/metricas_comparativas.csv",
) -> None:
    """
    Guarda la tabla comparativa en results/tables.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(path, index=False, encoding="utf-8-sig")