from src.environment import (
    HospitalEnvironment,
    HOSPITAL_MAP,
    START_POS,
    GOAL_POS,
    CHARGE_POS,
    MAX_BATTERY,
    W, Z, R,
)

SCENARIO_CHANGES = {
    "Base": {},
    "Congestionado": {
        Z: [
            (1, 2), (1, 3), (1, 6), (1, 7),
            (5, 2), (5, 3), (5, 6), (5, 7),
            (7, 2), (7, 3),
            (10, 10), (10, 11), (10, 12),
        ]
    },
    "Riesgo alto": {
        R: [
            (3, 4), (4, 6),
            (5, 8), (5, 9),
            (7, 8), (7, 9),
            (9, 10),
            (10, 11), (10, 12), (10, 13),
        ]
    },
    "Mixto": {
        Z: [
            (1, 2), (1, 3), (1, 6), (1, 7),
            (5, 2), (5, 3), (7, 2), (7, 3),
            (10, 10), (10, 11),
        ],
        R: [
            (3, 4), (4, 6),
            (5, 8), (5, 9),
            (7, 8), (9, 10),
            (10, 12), (10, 13),
        ],
    },
}

SCENARIO_NAMES = list(SCENARIO_CHANGES.keys())


def _clone_base_map():
    return [row.copy() for row in HOSPITAL_MAP]


def _apply_changes(hospital_map, changes):
    protected = {START_POS, GOAL_POS, CHARGE_POS}
    for cell_type, coords in changes.items():
        for row, col in coords:
            if (row, col) in protected:
                continue
            if hospital_map[row][col] == W:
                continue
            hospital_map[row][col] = cell_type
    return hospital_map


def make_scenario_map(scenario_name):
    """Devuelve una copia del mapa base con los cambios del escenario aplicados."""
    hospital_map = _clone_base_map()
    changes = SCENARIO_CHANGES.get(scenario_name, {})
    return _apply_changes(hospital_map, changes)


def create_env_for_scenario(scenario_name):
    """Crea un HospitalEnvironment con el mapa del escenario indicado."""
    env = HospitalEnvironment()
    env.map = make_scenario_map(scenario_name)
    env.rows = len(env.map)
    env.cols = len(env.map[0])
    env.max_battery = MAX_BATTERY
    env.reset()
    return env
