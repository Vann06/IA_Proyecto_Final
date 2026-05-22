import random
from enum import IntEnum

# ── Tipos de celda ─────────────────────────────────────────
W = 'W'   # Pared          — infranqueable
P = '.'   # Pasillo        — reward: -1
S = 'S'   # Inicio robot   — reward: -1
G = 'G'   # Destino        — reward: +100  (terminal)
C = 'C'   # Estación carga — reward: -1  (recarga batería)
R = 'R'   # Zona riesgo    — reward: -20
Z = 'Z'   # Zona congest.  — reward: -5

# ── Mapa del hospital (16 columnas × 12 filas) ─────────────
HOSPITAL_MAP = [
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],  # fila 0
    [W, S, P, P, Z, P, P, P, P, Z, P, P, P, P, P, W],  # fila 1  ← corredor norte
    [W, P, W, W, P, W, P, W, P, W, W, P, W, W, P, W],  # fila 2
    [W, P, W, Z, P, W, P, R, P, W, P, P, W, P, P, W],  # fila 3  ← cluster R1
    [W, P, W, P, W, Z, P, R, W, P, R, P, P, P, P, W],  # fila 4  ← cluster R1 + R2
    [W, P, P, P, P, C, P, P, P, P, R, P, P, P, P, W],  # fila 5  ← corredor central + R2
    [W, P, W, W, P, W, W, W, Z, W, W, P, W, W, P, W],  # fila 6
    [W, P, P, P, P, Z, P, W, P, P, P, Z, P, P, P, W],  # fila 7
    [W, P, W, W, P, W, W, W, P, R, R, W, P, W, P, W],  # fila 8  ← cluster R3
    [W, P, W, P, P, P, Z, P, W, R, P, P, P, W, P, W],  # fila 9  ← cluster R3
    [W, P, P, P, P, P, Z, P, P, P, P, P, P, P, G, W],  # fila 10 ← corredor sur
    [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W],  # fila 11
]

ROWS = len(HOSPITAL_MAP)        # 12
COLS = len(HOSPITAL_MAP[0])     # 16

# ── Rewards por tipo de celda ──────────────────────────────
CELL_REWARDS = {
    P: -1,
    S: -1,
    C: -1,
    Z: -5,
    R: -20,
    G: +100,
    W: None,
}

# ── Posiciones especiales ──────────────────────────────────
START_POS  = (1, 1)
GOAL_POS   = (10, 14)
CHARGE_POS = (5, 5)

# ── Batería ────────────────────────────────────────────────
MAX_BATTERY    = 30   # pasos máximos antes de agotarse
CHARGE_AMOUNT  = 10   # cuánto recarga cada acción CHARGE


# ── Acciones ───────────────────────────────────────────────
class Action(IntEnum):
    UP     = 0
    DOWN   = 1
    LEFT   = 2
    RIGHT  = 3
    STAY   = 4
    CHARGE = 5

N_ACTIONS = len(Action)

# Desplazamiento (delta_row, delta_col) para cada dirección de movimiento
MOVE_DELTAS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
    Action.STAY:  ( 0,  0),
}


class HospitalEnvironment:
    """
    GridWorld estocástico que representa un hospital.

    Estado: (row, col, battery)
      - row, col : posición del robot en el mapa
      - battery  : nivel de batería actual (0 … MAX_BATTERY)

    Las transiciones de movimiento son estocásticas:
      80 % → dirección deseada
      10 % → desvío perpendicular izquierdo
      10 % → desvío perpendicular derecho
    """

    def __init__(self):
        self.rows        = ROWS
        self.cols        = COLS
        self.map         = HOSPITAL_MAP
        self.max_battery = MAX_BATTERY
        self.state       = None          # se asigna en reset()

    def reset(self):
        """Reinicia el episodio. Devuelve el estado inicial."""
        self.state = (START_POS[0], START_POS[1], self.max_battery)
        return self.state

    def _actual_action(self, action):
        """
        Aplica estocasticidad: dada la acción deseada, devuelve la acción
        que realmente ocurre.

        Movimientos (UP/DOWN/LEFT/RIGHT):
          80% → dirección deseada
          10% → desvío perpendicular izquierdo
          10% → desvío perpendicular derecho
        STAY y CHARGE son siempre deterministas.
        """
        perp_left = {
            Action.UP:    Action.LEFT,
            Action.DOWN:  Action.RIGHT,
            Action.LEFT:  Action.DOWN,
            Action.RIGHT: Action.UP,
        }
        perp_right = {
            Action.UP:    Action.RIGHT,
            Action.DOWN:  Action.LEFT,
            Action.LEFT:  Action.UP,
            Action.RIGHT: Action.DOWN,
        }

        if action in (Action.STAY, Action.CHARGE):
            return action

        roll = random.random()
        if roll < 0.80:
            return action
        elif roll < 0.90:
            return perp_left[action]
        else:
            return perp_right[action]

    def _apply_move(self, row, col, action):
        """
        Calcula la celda destino al aplicar un movimiento.
        Si la celda destino es pared o está fuera del mapa, devuelve (row, col).
        """
        dr, dc = MOVE_DELTAS[action]
        new_row, new_col = row + dr, col + dc

        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return row, col
        if self.map[new_row][new_col] == W:
            return row, col

        return new_row, new_col

    
    def step(self, action):
        """
        Ejecuta una acción y avanza el entorno un paso.

        Retorna:
            next_state : (row, col, battery)
            reward     : recompensa de la celda destino (o penalización por fallo)
            done       : True si el episodio terminó
            info       : dict con detalles del paso
        """
        row, col, battery = self.state

        # 1. Estocasticidad: la acción real puede diferir de la pedida
        actual = self._actual_action(action)

        # 2. Calcular nueva posición y batería
        if actual == Action.CHARGE:
            new_row, new_col = row, col

            if (row, col) == CHARGE_POS:
                battery = min(battery + CHARGE_AMOUNT, self.max_battery)

            # CHARGE fuera de la estación no descuenta batería
        else:
            new_row, new_col = self._apply_move(row, col, actual)
            battery -= 1

        # 3. Recompensa según la celda donde quedó el robot
        cell_type = self.map[new_row][new_col]
        reward = CELL_REWARDS[cell_type]

        # 4. Condiciones de fin de episodio
        done = False
        success = False
        terminal_reason = None

        if (new_row, new_col) == GOAL_POS:
            done = True
            success = True
            terminal_reason = "goal"

        elif battery <= 0:
            battery = 0
            reward = -50
            done = True
            success = False
            terminal_reason = "no_battery"

        self.state = (new_row, new_col, battery)

        info = {
            "requested_action": Action(action).name,
            "actual_action": Action(actual).name,
            "cell_type": cell_type,
            "battery": battery,
            "success": success,
            "terminal_reason": terminal_reason,
        }

        return self.state, reward, done, info
    
    def get_possible_actions(self, state):
        """
        Devuelve la lista de acciones válidas para un estado dado.
        En estados terminales no hay acciones disponibles.
        CHARGE solo se incluye cuando el robot está en la estación de carga.
        """
        if self.is_terminal(state):
            return []

        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]

        row, col, _ = state
        if (row, col) == CHARGE_POS:
            actions.append(Action.CHARGE)

        return actions

    def is_terminal(self, state):
        """
        Devuelve True si el episodio terminó en este estado.
        Dos causas posibles: llegó al destino o se quedó sin batería.
        """
        row, col, battery = state
        return (row, col) == GOAL_POS or battery <= 0

    def get_all_states(self):
        """
        Devuelve todos los estados posibles del entorno.
        Value Iteration necesita esto para inicializar V(s) y recorrer estados.
        Solo incluye celdas que no sean pared.
        """
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map[row][col] != W:
                    for battery in range(0, self.max_battery + 1):
                        states.append((row, col, battery))
        return states
