# MedRoute: Robot médico asistente con IA

Proyecto final de **CC3045 – Inteligencia Artificial**.

Este proyecto simula un robot médico asistente en el área de urgencias de un hospital. El robot debe transportar sueros e insumos médicos desde un punto inicial hasta un destino, tomando decisiones en un ambiente estocástico con batería limitada, zonas congestionadas, zonas de riesgo y movimientos con incertidumbre.

El objetivo principal es comparar dos enfoques de toma de decisiones secuenciales:

- **Value Iteration**: método basado en modelo que utiliza el MDP completo y la ecuación de Bellman.
- **Q-Learning**: método de aprendizaje por refuerzo libre de modelo, donde el agente aprende por experiencia.

Ambos agentes se evalúan sobre el mismo entorno, con las mismas acciones, recompensas y métricas.

---

## 1. Descripción del problema

En un hospital, especialmente en el área de urgencias, el tiempo de entrega de insumos médicos puede ser importante para la atención de pacientes. Un robot asistente puede apoyar al personal trasladando sueros, medicamentos o materiales entre diferentes áreas.

Sin embargo, el hospital no es un entorno completamente predecible. Pueden existir:

- pasillos congestionados,
- zonas de riesgo o tránsito lento,
- paredes o rutas no disponibles,
- batería limitada,
- movimientos con incertidumbre.

Por esta razón, el problema no se trata únicamente de encontrar la ruta más corta, sino de tomar decisiones secuenciales bajo incertidumbre. Para modelarlo, se utiliza un **Proceso de Decisión de Markov (MDP)**.

---

## 2. Objetivo

Comparar el desempeño de **Value Iteration** y **Q-Learning** en un hospital simulado como GridWorld, evaluando cuál agente logra transportar insumos de forma más eficiente y segura.

### Objetivos específicos

- Modelar el hospital como un entorno discreto tipo GridWorld.
- Definir estados, acciones, recompensas y transiciones estocásticas.
- Implementar Value Iteration como método basado en modelo.
- Implementar Q-Learning como método de aprendizaje por experiencia.
- Evaluar ambos agentes con las mismas métricas.
- Analizar el proceso iterativo de mejora de Q-Learning mediante ajuste de hiperparámetros.

---

## 3. Conceptos de IA aplicados

| Concepto | Aplicación en el proyecto |
|---|---|
| Agente inteligente | El robot médico decide qué acción tomar en cada estado. |
| Ambiente estocástico | Las acciones de movimiento no siempre producen el resultado esperado. |
| MDP | El problema se modela con estados, acciones, recompensas y transiciones. |
| Propiedad de Markov | El siguiente estado depende del estado actual y de la acción aplicada. |
| Política | Estrategia que indica qué acción tomar en cada estado. |
| Value Iteration | Calcula una política usando el modelo completo del entorno. |
| Q-Learning | Aprende una política mediante interacción y actualización de una Q-table. |
| Exploración vs explotación | Q-Learning alterna entre probar acciones nuevas y usar lo aprendido. |

---

## 4. Entorno: `HospitalEnvironment`

El hospital se modela como una grilla de **12 × 16 celdas**.

| Símbolo | Significado | Recompensa |
|---|---|---:|
| `S` | Inicio del robot | -1 |
| `.` | Pasillo normal | -1 |
| `W` | Pared infranqueable | — |
| `Z` | Zona congestionada | -5 |
| `R` | Zona de riesgo | -20 |
| `C` | Estación de carga | -1 |
| `G` | Destino / entrega exitosa | +100 |

### Estado

El estado del robot se representa como:

```text
(row, col, battery)
```

Donde:

- `row`: fila actual del robot.
- `col`: columna actual del robot.
- `battery`: nivel actual de batería.

### Acciones disponibles

```text
UP, DOWN, LEFT, RIGHT, STAY, CHARGE
```

La acción `STAY` mantiene al robot en la misma posición, pero en la implementación actual también consume batería, simulando que el robot permanece encendido y operativo durante ese paso.

La acción `CHARGE` solo recarga batería si el robot está en la estación de carga.

### Transiciones estocásticas

Para acciones de movimiento:

| Resultado | Probabilidad |
|---|---:|
| Se ejecuta la acción deseada | 80% |
| Desvío perpendicular izquierdo | 10% |
| Desvío perpendicular derecho | 10% |

Las acciones `STAY` y `CHARGE` son deterministas.

### Condiciones terminales

Un episodio termina cuando:

- el robot llega al destino `G`,
- o se queda sin batería.

---

## 5. Estructura del repositorio

```text
IA_Proyecto_Final/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── environment.py        # Entorno hospitalario tipo GridWorld
│   ├── value_iteration.py    # Agente Value Iteration
│   ├── q_learning.py         # Agente Q-Learning
│   └── metrics.py            # Métricas y comparación de agentes
│
├── notebooks/
│   └── main.ipynb            # Flujo completo del proyecto
│
├── tests/
│   ├── test_environment.py   # Pruebas del entorno
│   └── test_q_learning.py    # Pruebas de Q-Learning
│
├── results/
│   ├── plots/                # Gráficas generadas
│   └── tables/               # Tablas generadas
│
└── docs/
    ├── informe_final.pdf
    └── presentacion.pdf
```


---

## 6. Instalación

Requiere **Python 3.10 o superior**.

### 6.1 Clonar el repositorio

```bash
git clone https://github.com/Vann06/IA_Proyecto_Final.git
cd IA_Proyecto_Final
```

### 6.2 Crear entorno virtual

En Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 6.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 7. Ejecución

### 7.1 Ejecutar pruebas

```bash
pytest tests/
```

Resultado esperado:

```text
66 passed
```

Las pruebas validan:

- posición inicial,
- consumo de batería,
- colisiones con paredes,
- estación de carga,
- recompensas por celda,
- estados terminales,
- actualización de Q-Learning,
- entrenamiento y política aprendida.

### 7.2 Ejecutar notebook principal

```bash
jupyter notebook notebooks/main.ipynb
```

El notebook realiza:

1. validación de pruebas,
2. visualización del mapa del hospital,
3. ejecución de Value Iteration,
4. entrenamiento de Q-Learning,
5. comparación de métricas,
6. gráficas de resultados,
7. pruebas adicionales con Q-Learning,
8. análisis por escenarios hospitalarios.

---

## 8. Algoritmos implementados

### 8.1 Value Iteration

Value Iteration utiliza el modelo completo del MDP para calcular el valor de cada estado mediante la ecuación de Bellman. Luego extrae una política que indica la mejor acción para cada estado.

Características:

- conoce las probabilidades de transición,
- conoce las recompensas,
- calcula valores `V(s)`,
- deriva una política óptima o cercana a la óptima.

### 8.2 Q-Learning

Q-Learning aprende una política mediante interacción con el entorno. El agente actualiza una tabla `Q(s,a)` según la recompensa obtenida y el mejor valor futuro estimado.

Características:

- no conoce directamente el modelo de transición,
- aprende por episodios,
- utiliza exploración y explotación,
- depende de hiperparámetros como `alpha`, `gamma` y `epsilon`.

---

## 9. Métricas de evaluación

| Métrica | Descripción |
|---|---|
| Porcentaje de entregas exitosas | Mide cuántas misiones llegaron al destino. |
| Tiempo promedio de entrega | Cantidad promedio de pasos en entregas exitosas. |
| Consumo promedio de batería | Batería utilizada durante los episodios. |
| Recompensa acumulada promedio | Calidad global de la política. |
| Fallos de misión | Episodios donde el robot no completó la entrega. |

---

## 10. Resultados principales

### 10.1 Comparación inicial

| Agente | Entregas exitosas | Tiempo promedio | Batería promedio | Recompensa promedio | Fallos |
|---|---:|---:|---:|---:|---:|
| Value Iteration | 90.0% | 28.98 | 26.00 | 48.60 | 10 |
| Q-Learning base | 53.0% | 33.42 | 29.61 | -14.30 | 47 |

Value Iteration obtuvo mejor desempeño inicial porque conoce el modelo completo del entorno. Q-Learning logró completar misiones, pero con más fallos y menor recompensa promedio.

### 10.2 Evolución de Q-Learning

| Configuración | Entregas exitosas | Recompensa promedio | Fallos |
|---|---:|---:|---:|
| Q-Learning inicial | 0.0% | -83.67 | 100 |
| Q-Learning base ajustado | 53.0% | -14.30 | 47 |
| Q-Learning con más episodios | 67.0% | 11.35 | 33 |
| Q-Learning con gamma alto | 77.0% | 25.17 | 23 |
| Value Iteration | 90.0% | 48.60 | 10 |

La mejor configuración de Q-Learning alcanzó **77% de entregas exitosas**, mostrando una mejora importante respecto a las primeras pruebas. Esto demuestra que el agente sí aprendió una política funcional mediante experiencia, aunque no superó a Value Iteration.

---

## 11. Análisis de resultados

Los resultados muestran que **Value Iteration fue el método más estable y eficiente**. Esto es esperable porque el algoritmo conoce el modelo completo del entorno: estados, acciones, recompensas y probabilidades de transición.

Q-Learning inició con bajo desempeño, pero mejoró al aumentar el entrenamiento y ajustar el factor de descuento `gamma`. En este problema, la recompensa más importante ocurre al final de la misión, cuando el robot llega al destino. Por eso, una configuración con `gamma` más alto permitió valorar mejor las recompensas futuras.

La comparación permite observar dos enfoques distintos:

| Enfoque | Característica | Resultado observado |
|---|---|---|
| Value Iteration | Planeación basada en modelo | Mayor estabilidad, más entregas y menos fallos. |
| Q-Learning | Aprendizaje por experiencia | Mejora progresiva, pero requiere más entrenamiento. |

---

## 12. Escenarios adicionales

Además del mapa base, el notebook incluye escenarios alternativos del hospital:

| Escenario | Descripción |
|---|---|
| Base | Hospital original. |
| Congestionado | Se agregan más zonas de congestión. |
| Riesgo alto | Se agregan más zonas de riesgo. |
| Mixto | Combina mayor congestión y riesgo. |

Estos escenarios permiten evaluar la robustez de los agentes ante cambios en el hospital.

---

## 13. Archivos generados

Al ejecutar el notebook se generan archivos en:

```text
results/
├── plots/
│   ├── mapa_hospital.png
│   ├── entregas_exitosas.png
│   ├── recompensa_promedio.png
│   ├── fallos_mision.png
│   ├── consumo_bateria.png
│   ├── tiempo_entrega.png
│   └── q_learning_rewards.png
│
└── tables/
    ├── metricas_comparativas.csv
    ├── detalle_value_iteration.csv
    ├── detalle_q_learning.csv
    └── ganador_por_metrica.csv
```

Dependiendo del notebook usado, también pueden generarse gráficas y tablas adicionales de escenarios.

## 15. Conclusión

El proyecto demuestra que un problema real de logística hospitalaria puede modelarse como un MDP. En este entorno, **Value Iteration fue más estable y eficiente** porque conoce el modelo del hospital y calcula una política usando Bellman.

**Q-Learning no superó a Value Iteration**, pero sí mostró aprendizaje progresivo. Pasó de un rendimiento inicial bajo a una mejor configuración con 77% de entregas exitosas. Esto demuestra que el agente puede aprender mediante experiencia, aunque requiere más episodios y ajuste de parámetros.

En conclusión, si el modelo del entorno se conoce, Value Iteration es una alternativa más directa y estable. Si el modelo no se conoce, Q-Learning es útil porque permite aprender desde la interacción.

---

## 16. Referencias

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- Documentación oficial de Python.
- Documentación oficial de Pandas.
- Documentación oficial de Matplotlib.
- Material del curso de Inteligencia Artificial.

