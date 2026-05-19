# Robot Médico IA

Proyecto final de **CC3045 – Inteligencia Artificial** (UVG).

Comparación de dos enfoques de decisión secuencial para un robot médico que opera en un hospital estocástico modelado como `GridWorld`:

- **Value Iteration** — planificación basada en modelo (Bellman).
- **Q-Learning** — aprendizaje por refuerzo libre de modelo (ε-greedy).

Ambos agentes comparten el mismo entorno, las mismas recompensas y las mismas reglas de transición. La comparación se hace bajo métricas idénticas para que el contraste sea justo.

---

## 1. Objetivo

Determinar cuál enfoque entrega mejor desempeño en un hospital discreto cuando:

- El robot conoce el modelo completo del entorno → se espera que **Value Iteration** sea óptimo.
- El robot debe aprender por interacción → se evalúa qué tan buena es la política aprendida por **Q-Learning**.

Métricas evaluadas:

| Métrica | Descripción |
|---|---|
| Pasos promedio hasta el goal | Eficiencia del recorrido |
| Consumo promedio de batería | Eficiencia energética |
| % entregas exitosas | Confiabilidad |
| Recompensa acumulada promedio | Calidad global de la política |
| Fallos de misión | Robustez |

---

## 2. Entorno: `HospitalEnvironment`

Hospital de **12 × 16 celdas** con los siguientes tipos:

| Símbolo | Significado | Recompensa |
|---|---|---:|
| `S` | Inicio | -1 |
| `.` | Pasillo | -1 |
| `W` | Pared | — (infranqueable) |
| `Z` | Zona congestionada | -5 |
| `R` | Zona de riesgo | -20 |
| `C` | Estación de carga | -1 |
| `G` | Destino (terminal) | +100 |

**Estado:** `(row, col, battery)` con `battery ∈ [0, 30]`.

**Acciones:** `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `CHARGE`.

**Transiciones estocásticas** (movimiento):

| Resultado | Probabilidad |
|---|---:|
| Acción deseada | 0.80 |
| Desvío perpendicular izquierdo | 0.10 |
| Desvío perpendicular derecho | 0.10 |

`STAY` y `CHARGE` son deterministas. `CHARGE` solo recarga (`+10`) en `CHARGE_POS = (5, 5)`.

**Terminales:** llegar al goal `(10, 14)` o quedarse sin batería (penalización `-50`).

---

## 3. Estructura del repositorio

```text
IA_Proyecto_Final/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── plan_equipo_robot_medico_IA_detallado.md
│
├── src/
│   ├── environment.py        # HospitalEnvironment (GridWorld estocástico)
│   ├── value_iteration.py    # ValueIterationAgent
│   ├── q_learning.py         # QLearningAgent
│   └── metrics.py            # Evaluación y comparación de agentes
│
├── notebooks/
│   └── main.ipynb            # Flujo completo: entrenamiento + evaluación + gráficas
│
├── tests/
│   └── test_environment.py   # Validación del entorno
│
└── results/
    ├── plots/                # Figuras generadas
    └── tables/               # Tablas CSV
```

---

## 4. Estado de implementación

| Módulo | Estado | Responsable |
|---|---|---|
| `src/environment.py` | Implementado | Integrante 1 |
| `tests/test_environment.py` | Implementado | Integrante 1 |
| `src/value_iteration.py` | **Implementado** | Integrante 2 |
| `src/q_learning.py` | Pendiente | Integrante 3 |
| `src/metrics.py` | Pendiente | Integrante 4 |
| `notebooks/main.ipynb` | Pendiente | Integrante 4 |

### 4.1 `ValueIterationAgent` (terminado)

Implementación de Value Iteration con modelo de transición reconstruido a partir de la lógica determinista de `HospitalEnvironment` (no se invoca el muestreo aleatorio de `step`).

API pública:

```python
from src.environment import HospitalEnvironment
from src.value_iteration import ValueIterationAgent

env = HospitalEnvironment()
agent = ValueIterationAgent(env, gamma=0.9, theta=1e-3)
agent.run(verbose=True)        # iteración hasta convergencia
action = agent.get_action(state)
```

| Método | Descripción |
|---|---|
| `run(verbose=False)` | Aplica Bellman hasta `Δ < theta`; guarda `self.V` y llama a `extract_policy`. |
| `compute_q_value(state, action)` | Calcula `Q(s,a) = Σ P(s'\|s,a)[R + γ V(s')]`. |
| `extract_policy()` | Política greedy a partir de `V`. |
| `get_action(state)` | Acción recomendada (`None` si estado terminal). |

Smoke test (gamma=0.9, theta=1e-3): converge en **52 iteraciones**, política inicial `DOWN`, rollout greedy completa misión con reward acumulado ≈ **+71**.

---

## 5. Instalación

Requiere **Python ≥ 3.10**.

```bash
git clone <repo-url>
cd IA_Proyecto_Final

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```

Dependencias: `numpy`, `matplotlib`, `pandas`, `jupyter`, `ipykernel`.

---

## 6. Ejecución

### 6.1 Notebook principal

```bash
jupyter notebook notebooks/main.ipynb
```

### 6.2 Uso directo desde Python

```python
from src.environment import HospitalEnvironment
from src.value_iteration import ValueIterationAgent

env = HospitalEnvironment()
vi = ValueIterationAgent(env, gamma=0.9, theta=1e-3)
vi.run(verbose=True)

state = env.reset()
done = False
total_reward = 0
while not done:
    action = vi.get_action(state)
    state, reward, done, info = env.step(action)
    total_reward += reward
print("Reward total:", total_reward)
```

### 6.3 Tests

```bash
pytest tests/
```

---

## 7. Distribución del equipo

| Integrante | Rol | Archivos |
|---|---|---|
| 1 | Entorno y pruebas | `src/environment.py`, `tests/test_environment.py` |
| 2 | Value Iteration | `src/value_iteration.py` |
| 3 | Q-Learning | `src/q_learning.py` |
| 4 | Métricas, notebook, README | `src/metrics.py`, `notebooks/main.ipynb`, `results/`, `README.md` |

---

## 8. Reglas del proyecto

- **Un solo entorno:** ningún agente puede modificar ni duplicar `HospitalEnvironment`. La comparación solo es justa si ambos algoritmos usan exactamente el mismo entorno, recompensas y reglas.
- **Mismas métricas:** los dos agentes se evalúan con `src/metrics.py` bajo idénticas condiciones.
- **Resultados versionados:** tablas y figuras finales se guardan en `results/`.

---

## 9. Referencias

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed. (MIT Press, 2018).
