# RL ConvNet: Reinforcement Learning with Spatial Kernels

Un framework de **aprendizaje por refuerzo** que utiliza **kernels convolucionales 3×3** para entrenar políticas en grillas 2D con orientaciones múltiples. El agente aprende a navegar desde un punto de inicio a una meta evitando obstáculos.

**Estado**: ✅ Completamente funcional con pipeline automatizado (Feb 2026)

---

## 🎯 Características Principales

- **Algoritmo**: Iteración de valores con convergencia adaptativa
- **Espacio de estados**: Grid 100×100 con hasta 8 orientaciones (configurable)
- **Espacio de acciones**: 6 acciones posibles/orientación (3 permitidas por defecto)
- **Núcleo**: Kernels 3×3 para agregación espacial de valores de vecinos
- **Generación de mapas**: Procedural con obstáculos suavizados y caminos garantizados
- **Recompensas inteligentes**: Combina seguridad (distancia a obstáculos) + objetivo (atracción a meta)
- **Simulación**: Argmax determinista o softmax con temperatura configurable
- **Pipeline completo**: Generación + entrenamiento + simulación en un solo comando
- **Reproducibilidad**: Nombres de carpetas contienen todos los parámetros

---

## 🚀 Inicio Rápido

### Pipeline Completo en Un Comando

```bash
# Compilar solver (solo una vez)
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm

# Generar mapa + entrenar + simular
python test_pipeline.py --density 0.4 --seed 42 --start 10,10 --goal 90,90 \
  --tol 1e-3 --k-max 5000 --action all --goal-radius 1
```

**¿Qué hace esto?**
1. Genera mapa en `sim_maps/map_d0.4_s42_sm3_cr2/`
2. Entrena política usando value iteration
3. Simula trayectoria y guarda `policy_path.png` en carpeta del mapa

**Resultado esperado:**
```
============================================================
Generating map: density=0.4, seed=42, start=10,10, goal=90,90
============================================================
  ✓ Generated: map_d0.4_s42_sm3_cr2

============================================================
Training on generated map
============================================================
🔧 Training policy...
  Policy saved to: policy.txt and sim_maps/map_d0.4_s42_sm3_cr2/policy.txt

🎮 Simulating policy...
  Auto-detected policy from: sim_maps\map_d0.4_s42_sm3_cr2\policy.txt
  Saved rollout plot to sim_maps\map_d0.4_s42_sm3_cr2\policy_path.png
  Simulation stop reason: reached-goal

✅ Pipeline complete!
```

---

## 📦 Estructura del Proyecto

```
RISCVsummit/
├── README.md                        ← Este archivo (inicio rápido)
├── README_TESTING.md                ← Guía completa (todos los parámetros)
├── RESTRUCTURING_SUMMARY.md         ← Cambios recientes
│
├── rl_convnet_simple.c              ← Solver (value iteration en C)
├── rl_convnet_simple.exe            ← Ejecutable compilado
│
├── sim/                             ← Generación de mapas
│   ├── map_generator.py             ← Crea mapas → carpetas auto-nombradas
│   └── map_visualize.py             ← Visualiza mapas
│
├── sim_maps/                        ← Base de datos de mapas
│   ├── map_d0.1_s55_sm1_cr1/        ← Móvil (10% obstáculos)
│   │   ├── obstacles.csv
│   │   ├── reward.csv
│   │   ├── start_goal.csv
│   │   └── obstacles.png            ← Auto-generado
│   │
│   ├── map_d0.35_s42_sm3_cr2/       ← Estándar (35% obstáculos)
│   │   └── ... (ídem)
│   │
│   └── map_d0.5_s100_sm4_cr1/       ← Difícil (50% obstáculos)
│       └── ... (ídem)
│
├── viz/                             ← Visualización de políticas
│   ├── plot_policy.py               ← Policy plotting + simulation
│   └── policy_plots/                ← Gráficos generados
│
├── policy.txt                       ← Política entrenada
└── test_pipeline.py                 ← Automatización train+simulate
```

---

## 📚 Documentación

| Archivo | Contenido |
|---------|----------|
| **Este README** | Intro rápida, inicio de 5 min |
| **README_TESTING.md** | Parámetros completos, ejemplos, troubleshooting |
| **RESTRUCTURING_SUMMARY.md** | Cambios en estructura, beneficios |

---

## 🔄 Flujos de Trabajo Comunes

### Flujo 1: Training rápido
```bash
# Generar mapa fácil
python sim/map_generator.py --density 0.1 --seed 1

# Entrenar
./rl_convnet_simple.exe --reward sim_maps/map_d0.1_s1_sm3_cr2/reward.csv

# Simular
python viz/plot_policy.py --simulate
```

### Flujo 2: Experimentos sistemáticos
```bash
# Generar suite de mapas
python sim/map_generator.py --density 0.2 --seed 100
python sim/map_generator.py --density 0.4 --seed 100
python sim/map_generator.py --density 0.6 --seed 100

# Entrenar todos
python test_pipeline.py --action all --all-maps
```

### Flujo 3: Análisis de convergencia
```bash
# Entrenar con diferentes tolerancias
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2 --tol 1e-2
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2 --tol 1e-4
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2 --tol 1e-6

# Comparar iteraciones de convergencia
```

### Flujo 4: Exploración vs explotación
```bash
# Determinista (mejor policy)
python viz/plot_policy.py --simulate --action-mode argmax

# Exploración suave
python viz/plot_policy.py --simulate --action-mode softmax --temperature 0.5

# Exploración agresiva
python viz/plot_policy.py --simulate --action-mode softmax --temperature 2.0
```

---

## ⚙️ Parámetros Clave

### Solver: `./rl_convnet_simple.exe`
```bash
--reward <path>              # CSV de recompensas (requerido)
--tol <float>                # Tolerancia convergencia (default: 1e-4)
--k-max <int>                # Max iteraciones (default: 1000)
--n-orient <int>             # Orientaciones: 1-8 (default: 8)
--n-actions <int>            # Acciones/orientación: 1-6 (default: 6)
--allowed-actions <int>      # Solo primeras N (default: 3)
```

### Map Generator: `python sim/map_generator.py`
```bash
--density <float>            # Obstáculos 0-1 (default: 0.35)
--seed <int>                 # Reproducibilidad (default: 42)
--smooth <int>               # Suavizado (default: 3)
--carve <int>                # Radio camino (default: 2)
--width <int>                # Ancho grid (default: 100)
--height <int>               # Alto grid (default: 100)
```

### Test Pipeline: `python test_pipeline.py`
```bash
--action {list|train|simulate|all}   # Operación
--map <nombre>                       # Mapa específico
--all-maps                           # Todos los mapas
--tol <float>                        # Tol. solver
--k-max <int>                        # Max iters solver
--action-mode {argmax|softmax}       # Selección acción
--temperature <float>                # Para softmax
```

---

## 📊 Nomenclatura de Mapas

Formato: `map_dX_sY_smZ_crW`

Ejemplo: `map_d0.35_s42_sm3_cr2`
- `d` = Densidad (0.35 → 35%)
- `s` = Seed (42 → reproducible)
- `sm` = Suavizado (3 → 3 pasadas)
- `cr` = Carve radius (2 → radio 2)

Esto hace que el nombre contenga **todos los parámetros** para reproducir el mapa.

---

## 📈 Algoritmo: Value Iteration con Kernels

```
Para cada iteración t:
   1. Aggregation:  Convolve V(t) con kernels 3x3 por dirección
   2. Backup:       V(t+1) ← R + γ * max_a aggregate(a)
   3. Convergence:  delta = max_s |V(t+1) - V(t)|
   4. Check:        si delta < tol → CONVERGED
                    si t >= k-max → MAXED

Política:    π(a|s) = exp(Q(a,s) - V(s)) / Z    (softmax)
```

---

## 🎮 Casos de Uso

| Caso | Comando |
|------|---------|
| Training básico | `./rl_convnet_simple.exe --reward sim_maps/.../reward.csv` |
| Convergencia rápida | `--tol 1e-2 --k-max 5000` |
| Convergencia precisa | `--tol 1e-6 --k-max 100000` |
| Simulación determinista | `python viz/plot_policy.py --simulate --action-mode argmax` |
| Simulación estocástica | `python viz/plot_policy.py --simulate --action-mode softmax --temperature 1.5` |
| Batch training | `python test_pipeline.py --action all --all-maps` |
| Listar mapas | `python test_pipeline.py --action list` |

---

## ✅ Checklist de Instalación

- [ ] Compilar: `gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm`
- [ ] Generar: `python sim/map_generator.py --density 0.35`
- [ ] Entrenar: `./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv`
- [ ] Simular: `python viz/plot_policy.py --simulate`
- [ ] Visualizar: abrir `policy_plots/policy_path.png`

**🎉 ¡Listo para experimentar!**

---

## 📖 Para Más Detalles

Consulta:
- **[README_TESTING.md](README_TESTING.md)** para guía completa con todos los parámetros
- **[RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md)** para entender los cambios recientes
- **Código fuente** para arquitectura interna

---

**Última actualización**: 12 Feb 2026  
**Versión**: 2.0 (con reorganización de carpetas y test pipeline)
