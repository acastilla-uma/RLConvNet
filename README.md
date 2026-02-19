# RL ConvNet: Reinforcement Learning with Spatial Kernels

Framework de aprendizaje por refuerzo que entrena políticas en grillas 2D con orientaciones múltiples. El agente aprende a navegar desde inicio a meta evitando obstáculos usando value iteration con kernels 3×3.

**Status**: ✅ Completamente funcional (Feb 2026)

---

## 🔄 Workflow del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. GENERACIÓN DE MAPA (map_generator.py)                      │
│     Entrada: --density, --seed, --smooth, --carve              │
│             --start, --goal, --goal-reward                     │
│     Salida:  sim_maps/map_dX_sY_smZ_crW/                       │
│             ├── obstacles.csv      (grid binario 0/1)          │
│             ├── obstacles.png      (visualización)             │
│             ├── reward.csv         (recompensas por celda)     │
│             └── start_goal.csv     (sx,sy,gx,gy)               │
│                                                                 │
│  2. ENTRENAMIENTO (rl_convnet_simple.exe)                      │
│     Entrada: reward.csv, --tol, --k-max                        │
│     Proceso: Value iteration con kernels 3×3                   │
│     Salida:  policy.txt            (Q-values 8×6×100×100)      │
│     Métricas: Iteraciones, convergencia, V(goal)               │
│                                                                 │
│  3. SIMULACIÓN (viz/plot_policy.py)                            │
│     Entrada: policy.txt, reward.csv, start_goal.csv            │
│             --action-mode, --goal-radius                       │
│     Proceso: Rollout desde inicio hasta:                       │
│             • Alcanza objetivo                                 │
│             • Choca con obstáculo                              │
│             • Detecta ciclo                                    │
│             • Alcanza max pasos                                │
│     Salida:  policy_path.png       (trayectoria visualizada)   │
│                                                                 │
│  4. BATCH EXPERIMENTS (batch_experiments.py)                   │
│     Entrada: --map, --k-max (lista), --tol (lista)            │
│     Proceso: Entrena múltiples configuraciones                 │
│     Salida:  map_XXX.html          (reporte interactivo)       │
│             • Tabla comparativa de experimentos                │
│             • Trayectorias visualizadas embebidas              │
│             • Mosaicos de políticas (opcional)                 │
│             • Notas editables persistentes                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes del Sistema

#### 📁 **map_generator.py** - Generador de Mapas
**Función:** Crea entornos de navegación 2D procedurales con obstáculos

**Entrada:**
- `density`: Porcentaje de obstáculos (0.0-1.0)
- `seed`: Semilla aleatoria para reproducibilidad
- `smooth`: Pasadas de suavizado para blobs coherentes
- `carve`: Radio para garantizar corredor libre inicio→meta
- `start/goal`: Posiciones inicial y objetivo (x,y)
- `goal-reward`: Recompensa alta en región del objetivo

**Salida:**
- `obstacles.csv`: Matriz 100×100 de 0 (libre) / 1 (obstáculo)
- `reward.csv`: Matriz 100×100 de valores flotantes:
  - Celdas con obstáculos: penalización negativa
  - Celdas libres: interpolación lineal según distancia a obstáculos
  - Región objetivo: boost alto para atraer política
- `start_goal.csv`: Una línea con `sx,sy,gx,gy`
- `obstacles.png`: Visualización con inicio (●) y meta (★)

**Algoritmo:**
1. Genera ruido aleatorio binario con `density`
2. Aplica `smooth` iteraciones de suavizado por conteo de vecinos
3. Dibuja corredor garantizado con radio `carve` usando A*
4. Construye mapa de recompensas híbrido (obstáculos + objetivo)

---

#### ⚙️ **rl_convnet_simple.c** - Solver de Value Iteration
**Función:** Entrena política óptima usando iteración de valores con kernels espaciales

**Entrada:**
- `reward.csv`: Mapa de recompensas (requerido)
- `--tol`: Tolerancia de convergencia (default: 1e-4)
- `--k-max`: Máximo de iteraciones (default: 1000)
- `--n-orient`: Orientaciones 1-8 (default: 8)
- `--n-actions`: Acciones por orientación 1-6 (default: 6)

**Salida:**
- `policy.txt`: Archivo de ~6MB con estructura:
  ```
  # iters=<N>
  <orient> <action> <y> <x> <Q-value>
  ```
  Contiene Q(s,a) para cada estado-acción

**Algoritmo (Value Iteration con Kernels):**
1. **Inicialización**: V(s) ← 0 para todos los estados
2. **Para cada iteración t hasta k-max:**
   - **Aggregation**: Para cada orientación y acción:
     ```
     aggregate[o,a,y,x] = Σ kernel[o,a,dy,dx] * V[y+dy, x+dx]
     ```
   - **Backup**: 
     ```
     Q[o,a,y,x] = reward[y,x] + γ * aggregate[o,a,y,x]
     V_new[o,y,x] = max_a Q[o,a,y,x]
     ```
   - **Convergencia**:
     ```
     delta = max_s |V_new[s] - V[s]|
     if delta < tol: CONVERGED
     ```
3. **Política**: Softmax sobre Q-values

**Kernels 3×3:** Definen transición espacial por acción:
- Acción 0 (avanzar): Mueve en dirección de orientación
- Acción 1/2 (girar): Rota ±45°
- Cada kernel codifica probabilidad de alcanzar celdas vecinas

**Características:**
- Convergencia adaptativa: Para cuando mejora < tol
- Timeout: Para si alcanza k-max iteraciones
- Métricas: Reporta V(goal), iteraciones, estado de convergencia

---

#### 📊 **plot_policy.py** - Visualización y Simulación
**Función:** Genera trayectorias simuladas y mosaicos de políticas

**Modos de Operación:**

**1. Simulación de Trayectoria** (`--simulate`)
- **Entrada:**
  - `policy.txt`: Política entrenada
  - `reward.csv`: Para detectar obstáculos
  - `start_goal.csv`: Auto-detecta inicio/meta
  - `--action-mode`: `argmax` (greedy) o `softmax` (estocástico)
  - `--goal-radius`: Tolerancia de llegada (0=exacto)
  - `--steps`: Máximo de pasos (default: 10000)

- **Proceso:**
  1. Comienza en `(start_x, start_y, orient=0)`
  2. En cada paso:
     - Lee Q-values de estado actual
     - Selecciona acción según `action-mode`
     - Aplica transición usando kernels
     - Detecta condiciones de parada
  3. Para cuando:
     - `reached-goal`: Distancia al objetivo ≤ goal-radius
     - `hit-obstacle`: Entra en celda con obstáculo
     - `cycle`: Revisita estado (pos + orient)
     - `max-steps`: Agota límite de pasos

- **Salida:**
  - `policy_path.png`: Trayectoria sobre mapa de obstáculos
    - Línea azul: Camino seguido
    - Punto verde: Inicio
    - Estrella roja: Objetivo
    - Título con razón de parada

**2. Mosaico de Políticas** (`--mosaic`)
- **Entrada:**
  - `policy.txt`: Política entrenada
  - `--arrows`: Superpone flechas direccionales
  - `--stride`: Muestreo de flechas (default: 5)
  
- **Salida:**
  - `policy_argmax_mosaic.png`: Heatmap de V(s) con flechas
    - Color: Valor de estado (azul=bajo, rojo=alto)
    - Flechas: Acción óptima por orientación

**Características:**
- **Auto-detección**: Busca `policy.txt` en carpeta del mapa
- **Flexibilidad**: Soporta grillas de cualquier tamaño
- **Debugging**: `--print-table` para ver mapeo de acciones

---

#### 🔬 **batch_experiments.py** - Experimentos en Batch
**Función:** Ejecuta múltiples configuraciones de entrenamiento y genera reporte HTML interactivo

**Entrada:**
- `--map`: Mapa existente o `--all-maps` para procesar todos
- `--k-max`: Lista de valores (ej: 50 100 1000 5000)
- `--tol`: Lista de tolerancias (ej: 1e-2 1e-3 1e-4)
- `--generate-mosaic`: Flag para incluir mosaicos de políticas
- `--goal-radius`: Radio de tolerancia para simulaciones

**Proceso:**
1. **Por cada combinación (map, tol, k-max):**
   - Ejecuta `rl_convnet_simple.exe` con parámetros
   - Captura métricas: iteraciones, convergencia, V(goal)
   - Simula trayectoria con `plot_policy.py`
   - Embebe imagen como base64 en memoria
   - (Opcional) Genera mosaic de política
   - Elimina archivos temporales

2. **Genera reporte HTML:**
   - Tabla comparativa de todos los experimentos
   - Visualizaciones embebidas (no archivos externos)
   - Sistema de notas persistentes (JavaScript)
   - Diseño responsive con CSS moderno

**Salida:**
- `sim_maps/<map>/<map>.html`: Reporte completo
  - **Tabla de experimentos:** tol, k-max, iters, convergencia
  - **Sección por experimento:**
    - Trayectoria simulada (siempre)
    - Mosaico de políticas (si --generate-mosaic)
    - Razón de detención de simulación
  - **Notas editables:** Se guardan en HTML descargado

**Características Especiales:**
- **Imágenes embebidas**: Todo en un archivo HTML portátil
- **Notas persistentes**: JavaScript permite añadir comentarios
  - Guarda modificaciones descargando nuevo HTML
  - Notas viajan con el reporte
- **Diseño profesional**: CSS con colores y layouts limpios
- **Auto-detección de solver**: Busca ejecutable automáticamente

**Ejemplo de uso completo:**
```bash
# Batch con 4 configuraciones + mosaicos
python batch_experiments.py \
  --map map_d0.7_s42_sm3_cr2 \
  --k-max 50 100 1000 5000 \
  --tol 1e-3 \
  --generate-mosaic \
  --goal-radius 1
```

---

#### 🚀 **test_pipeline.py** - Automatización del Pipeline
**Función:** Orquesta generación → entrenamiento → simulación en un solo comando

**Entrada:**
- `--action`:
  - `list`: Lista mapas disponibles
  - `train`: Solo entrenar en mapa existente
  - `simulate`: Solo simular en mapa existente
  - `all`: Pipeline completo
- `--map`: Especifica mapa existente (o genera nuevo con --density/--seed)
- `--all-maps`: Procesa todos los mapas en batch

**Proceso según acción:**

**`--action list`:**
- Escanea `sim_maps/`
- Muestra mapas con detalles (densidad, seed, etc.)

**`--action train`:**
1. Valida existencia de `reward.csv` en mapa
2. Auto-detecta solver ejecutable
3. Ejecuta `rl_convnet_simple.exe` con parámetros
4. Reporta métricas de convergencia

**`--action simulate`:**
1. Valida existencia de `policy.txt` en mapa
2. Ejecuta `plot_policy.py --simulate`
3. Genera `policy_path.png`

**`--action all`:**
1. Si `--density` y `--seed`: Genera nuevo mapa
2. Entrena política en mapa
3. Simula trayectoria
4. Reporta resumen completo

**Características:**
- **Auto-detección**: Encuentra solver y archivos necesarios
- **Manejo de errores**: Valida existencia de archivos antes de ejecutar
- **Batch processing**: `--all-maps` itera sobre todos
- **Timeouts**: Previene bloqueo en simulaciones largas
- **Parseo de salida**: Extrae métricas de stdout del solver

**Ejemplo workflow completo:**
```bash
# Generar mapa nuevo + entrenar + simular
python test_pipeline.py \
  --density 0.5 --seed 999 \
  --tol 1e-4 --k-max 10000 \
  --goal-radius 2 \
  --action all
```

---

### Flujo de Datos Entre Componentes

```
map_generator.py
    │
    ├─→ obstacles.csv ──┐
    ├─→ obstacles.png   │
    ├─→ reward.csv ─────┼─→ rl_convnet_simple.exe
    └─→ start_goal.csv ─┤        │
                        │        ├─→ policy.txt ──┐
                        │        │                │
                        └────────┼────────────────┼─→ plot_policy.py
                                 │                │        │
                                 └────────────────┘        ├─→ policy_path.png
                                                           └─→ policy_argmax_mosaic.png
                                                                      │
                                                                      ↓
                                                           batch_experiments.py
                                                                      │
                                                                      └─→ map_XXX.html
                                                                          (reporte completo)
```

---

## 🎯 Características

- Algoritmo: Iteración de valores con convergencia adaptativa
- Espacios: Grid 100×100 con 8 orientaciones (configurable)
- Generación de mapas: Procedural con obstáculos suavizados
- Pipeline completo: Generar + entrenar + simular en un comando
- Reproducibilidad: Nombres de carpetas contienen todos los parámetros

---

## 🚀 Inicio Rápido

### Setup
```bash
# Compilar solver (solo una vez)
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm

# Generar mapa + entrenar + simular (todo en uno)
python test_pipeline.py --density 0.4 --seed 42 --tol 1e-3 --k-max 5000 --action all
```

**Resultado**: Mapa en `sim_maps/map_d0.4_s42_sm3_cr2/` con política entrenada y trayectoria simulada.

### Trabajar con Mapas Existentes
```bash
# Entrenar en mapa existente
python test_pipeline.py --action train --map map_d0.4_s42_sm3_cr2 --tol 1e-3 --k-max 5000

# Simular en mapa existente
python test_pipeline.py --action simulate --map map_d0.4_s42_sm3_cr2

# Listar mapas disponibles
python test_pipeline.py --action list
```

---

## 📦 Estructura del Proyecto

```
RISCVsummit/
├── README.md                    ← Este archivo
├── rl_convnet_simple.c          ← Solver (value iteration en C)
├── rl_convnet_simple.exe        ← Ejecutable compilado
│
├── sim/
│   ├── map_generator.py         ← Generador de mapas
│   └── map_visualize.py         ← Visualización de mapas
│
├── sim_maps/                    ← Base de datos de mapas
│   ├── map_d0.1_s55_sm1_cr1/   ← Móvil (10% obstáculos)
│   ├── map_d0.35_s42_sm3_cr2/  ← Estándar
│   └── map_d0.5_s100_sm4_cr1/  ← Difícil (50% obstáculos)
│
├── viz/
│   ├── plot_policy.py           ← Visualización y simulación
│   └── policy_plots/            ← Gráficos generados
│
├── test_pipeline.py             ← Automatización train+simulate
├── batch_experiments.py         ← Experimentos en batch
└── utils.py                     ← Funciones compartidas
```

---

## ⚙️ Referencia Completa de Parámetros

### Pipeline Principal: `python test_pipeline.py`

#### Acciones
- `--action list`: Listar mapas disponibles (default)
- `--action train`: Entrenar política en mapa existente
- `--action simulate`: Simular política en mapa existente
- `--action all`: Pipeline completo (generar + entrenar + simular)

#### Parámetros de Generación de Mapas

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--density` | float | - | Densidad de obstáculos (0-1), **requerido con --seed** |
| `--seed` | int | - | Semilla aleatoria para reproducibilidad, **requerido con --density** |
| `--smooth` | int | 3 | Pasadas de suavizado para blobs de obstáculos |
| `--carve` | int | 2 | Radio para corredor libre garantizado |
| `--start` | str | "5,5" | Posición inicial como 'x,y' |
| `--goal` | str | "94,94" | Posición objetivo como 'x,y' |

**Nombrado de carpetas:** `map_d{density}_s{seed}_sm{smooth}_cr{carve}`

**Ejemplo:**
```bash
python test_pipeline.py --density 0.5 --seed 999 --smooth 5 --carve 3 --action all
# Crea: sim_maps/map_d0.5_s999_sm5_cr3/
```

#### Parámetros del Solver

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--tol` | float | 1e-4 | Tolerancia de convergencia (para cuando max_delta < tol) |
| `--k-max` | int | 50000 | Máximo de iteraciones antes de timeout |

**Ejemplos:**

Entrenamiento rápido (puede no converger completamente):
```bash
--tol 1e-2 --k-max 1000
```

Entrenamiento de alta calidad:
```bash
--tol 1e-6 --k-max 100000
```

#### Parámetros de Simulación

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--steps` | int | 10000 | Máximo de pasos de simulación |
| `--action-mode` | str | "argmax" | Selección de acción: `argmax` (greedy) o `softmax` (estocástico) |
| `--temperature` | float | 1.0 | Temperatura para softmax (mayor = más exploración) |
| `--goal-radius` | int | 0 | Radio de tolerancia del objetivo (0 = coincidencia exacta) |

**Ejemplos:**

Trayectoria greedy (determinista):
```bash
--action-mode argmax --goal-radius 1
```

Trayectoria exploratoria (estocástica):
```bash
--action-mode softmax --temperature 0.5 --goal-radius 2
```

#### Trabajar con Mapas Existentes

Entrenar en mapa existente:
```bash
python test_pipeline.py --action train --map map_d0.4_s42_sm3_cr2 --tol 1e-3 --k-max 5000
```

Simular en mapa existente:
```bash
python test_pipeline.py --action simulate --map map_d0.4_s42_sm3_cr2 --goal-radius 1
```

Procesar todos los mapas en batch:
```bash
python test_pipeline.py --action all --all-maps --tol 1e-3 --k-max 1000
```

---

### Generación Manual de Mapas: `python sim/map_generator.py`

Generar mapas sin entrenar:

```bash
python sim/map_generator.py --density 0.35 --seed 42 --smooth 3 --carve 2 \
  --start 10,10 --goal 85,85 --goal-reward 10.0 --goal-radius 2
```

#### Parámetros Avanzados

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--width` | int | 100 | Ancho del mapa |
| `--height` | int | 100 | Alto del mapa |
| `--obstacle-penalty` | float | -1.0 | Recompensa para celdas con obstáculos |
| `--free-reward` | float | 0.0 | Recompensa base para celdas libres |
| `--reward-min` | float | 0.0 | Recompensa mínima para celdas lejos de obstáculos |
| `--reward-max` | float | 1.0 | Recompensa máxima para celdas cerca de obstáculos |
| `--goal-reward` | float | 5.0 | Alta recompensa en/cerca del objetivo (atrae política) |
| `--goal-radius` | int | 1 | Radio alrededor del objetivo con boost de recompensa |
| `--out` | str | "sim_maps" | Directorio de salida |

#### Cómo Funciona la Recompensa

1. **Celdas con obstáculos:** Penalización fija (`--obstacle-penalty = -1.0`)
2. **Celdas libres:** Interpolación lineal basada en distancia a obstáculos:
   - Lejos de obstáculos → `--reward-max`
   - Cerca de obstáculos → `--reward-min`
3. **Región del objetivo:** Recompensa fuerte (`--goal-reward = 5.0`) dentro de `--goal-radius` del objetivo

Esto crea una **preferencia por rutas seguras** (evita obstáculos) mientras **garantiza convergencia al objetivo**.

**Archivos generados en carpeta del mapa:**
- `obstacles.csv`: Grid binario (0=libre, 1=obstáculo)
- `obstacles.png`: Visualización con inicio (verde) y objetivo (estrella roja)
- `reward.csv`: Recompensa flotante por celda
- `start_goal.csv`: Línea única con `sx,sy,gx,gy`
- `policy.txt`: Generado después del entrenamiento
- `policy_path.png`: Generado después de la simulación

---

### Entrenamiento Manual: `./rl_convnet_simple.exe`

Compilar y ejecutar solver directamente:

```bash
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm

./rl_convnet_simple.exe --reward sim_maps/map_d0.4_s42_sm3_cr2/reward.csv \
  --tol 1e-3 --k-max 5000
```

#### Parámetros del Solver

```bash
--reward <path>              # CSV de recompensas (requerido)
--tol <float>                # Tolerancia convergencia (default: 1e-4)
--k-max <int>                # Max iteraciones (default: 1000)
--n-orient <int>             # Orientaciones: 1-8 (default: 8)
--n-actions <int>            # Acciones/orientación: 1-6 (default: 6)
--allowed-actions <int>      # Solo primeras N (default: 3)
```

**Características clave:**
- Kernels de convolución espacial 3×3 por orientación
- Detección de convergencia adaptativa
- Guarda política en raíz y en carpeta del mapa

**Mensajes de salida:**
```
[t=500] max_delta=0.012345
Converged at iteration 500 with tol=0.001
Policy saved to: policy.txt and sim_maps/map_d0.4_s42_sm3_cr2/policy.txt
```

---

### Visualización Manual: `python viz/plot_policy.py`

#### Simular Trayectoria

```bash
python viz/plot_policy.py --simulate \
  --reward sim_maps/map_d0.4_s42_sm3_cr2/reward.csv \
  --action-mode argmax --goal-radius 1 \
  --out-dir sim_maps/map_d0.4_s42_sm3_cr2
```

**Auto-detección:** Encuentra automáticamente `policy.txt` en carpeta del mapa (no necesita `--input`)

#### Parámetros de Visualización

| Parámetro | Descripción |
|-----------|-------------|
| `--input` | Ruta a policy.txt (auto-detectado si se omite) |
| `--reward` | Ruta a reward.csv para simulación |
| `--orient` | Índice de orientación (0-7) |
| `--action` | Índice de acción (0-5) |
| `--arrows` | Superponer flechas de dirección |
| `--stride` | Muestreo de flechas (salto) |
| `--start-x`, `--start-y` | Inicio de simulación (por defecto desde start_goal.csv) |
| `--start-orient` | Orientación inicial |
| `--steps` | Máximo de pasos de simulación |
| `--action-mode` | `argmax` (greedy) o `softmax` (estocástico) |
| `--temperature` | Temperatura softmax |
| `--goal-radius` | Radio de tolerancia del objetivo |
| `--out-dir` | Directorio de salida |
| `--show` | Mostrar ventana de plot |
| `--print-table` | Imprimir tabla de direcciones de acciones |

---

## � Ejemplos Completos de Flujos de Trabajo

### Ejemplo 1: Mapa de Alta Densidad con Navegación Segura

```bash
python test_pipeline.py \
  --density 0.6 --seed 777 \
  --start 5,5 --goal 95,95 \
  --smooth 5 --carve 3 \
  --tol 1e-4 --k-max 20000 \
  --goal-radius 2 \
  --action all
```

**Qué hace:**
- Genera mapa denso de obstáculos (60%)
- Corredores seguros amplios (carve=3, smooth=5)
- Entrenamiento de alta calidad (tol=1e-4)
- Región objetivo tolerante (radio=2)

### Ejemplo 2: Pruebas Rápidas

```bash
python test_pipeline.py \
  --density 0.3 --seed 123 \
  --tol 1e-2 --k-max 500 \
  --goal-radius 1 \
  --action all
```

**Qué hace:**
- Mapa simple (30% obstáculos)
- Entrenamiento rápido (pocas iteraciones)
- Workflow de validación rápida

### Ejemplo 3: Trayectorias Exploratorias

```bash
python test_pipeline.py \
  --density 0.4 --seed 42 \
  --start 10,10 --goal 80,80 \
  --tol 1e-3 --k-max 10000 \
  --action-mode softmax --temperature 0.3 \
  --goal-radius 1 \
  --action all
```

**Qué hace:**
- Selección estocástica de acciones (softmax)
- Temperatura baja = mayormente óptimo con pequeña exploración
- Produce trayectorias variadas en ejecuciones repetidas

### Ejemplo 4: Procesamiento en Batch

Generar múltiples mapas con diferentes semillas:

```bash
for seed in 100 200 300 400 500; do
  python test_pipeline.py \
    --density 0.4 --seed $seed \
    --tol 1e-3 --k-max 5000 \
    --goal-radius 1 \
    --action all
done
```

Procesar todos los mapas existentes:

```bash
python test_pipeline.py --action all --all-maps --tol 1e-3 --k-max 5000 --goal-radius 1
```

---

## 🔧 Resolución de Problemas

### "Simulation stop reason: cycle"

**Causa:** El agente revisitó el mismo estado (posición + orientación) antes de llegar al objetivo.

**Soluciones:**
1. Aumentar `--goal-radius` (ej., de 0 a 1 o 2)
2. Usar `--action-mode softmax --temperature 0.5` para romper ciclos
3. Aumentar `--goal-reward` en generación de mapa (ej., de 5.0 a 10.0)
4. Reducir `--tol` o aumentar `--k-max` para mejor convergencia

### "Simulation stop reason: hit-obstacle"

**Causa:** Calidad de política insuficiente o mapa muy difícil.

**Soluciones:**
1. Disminuir `--tol` (ej., de 1e-2 a 1e-4)
2. Aumentar `--k-max` (ej., de 1000 a 10000)
3. Aumentar radio `--carve` para corredores más amplios
4. Aumentar pasadas `--smooth` para obstáculos más limpios
5. Reducir `--density` para menos obstáculos

### "Warning: hit k-max without reaching tol"

**Causa:** Iteraciones de entrenamiento agotadas antes de convergencia.

**Soluciones:**
1. Aumentar `--k-max` (ej., duplicarlo)
2. Aumentar `--tol` si la precisión no es crítica
3. Verificar dificultad del mapa (densidad muy alta puede ser irresoluble)

### La Política No Llega al Objetivo

**Causa:** La recompensa no atrae suficientemente hacia el objetivo.

**Soluciones:**
1. Aumentar `--goal-reward` al generar mapa (ej., 10.0 o 20.0)
2. Aumentar `--goal-radius` para crear región atractiva más grande
3. Asegurar que la generación del mapa incluye recompensa de objetivo (automático en versión nueva)

---

## 💡 Consejos para Buenos Resultados

1. **Empezar simple:** Usar baja densidad (0.2-0.4) y entrenamiento rápido (k-max=1000) para pruebas
2. **Región objetivo:** Siempre usar `--goal-radius 1` o mayor para evitar problemas de ciclos
3. **Rutas seguras:** Default `--goal-reward 5.0` balancea seguridad y atracción al objetivo
4. **Convergencia:** Para resultados de calidad de publicación, usar `--tol 1e-5` y `--k-max 50000`
5. **Reproducibilidad:** Siempre especificar `--seed` para resultados deterministas
6. **Visualización:** Usar `--arrows --stride 5` para ver campos de dirección de política

---

## �📊 Nomenclatura de Mapas

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

## ⚡ Notas de Rendimiento

- **Generación de mapas:** < 1 segundo
- **Entrenamiento:** 1-60 segundos dependiendo de k-max y convergencia
- **Simulación:** < 1 segundo
- **Pipeline completo:** Típicamente 5-30 segundos

**Uso de memoria:**
- Archivo de política: ~6 MB para grid 100×100 con 8 orientaciones y 6 acciones
- Todos los archivos por mapa: ~12-15 MB

---

## 📝 Funciones Avanzadas

### Funciones de Recompensa Personalizadas

Editar `build_reward_map()` en [sim/map_generator.py](sim/map_generator.py) para implementar:

- Recompensas basadas en distancia al objetivo (en lugar de obstáculos)
- Escenarios multi-objetivo
- Costos de terreno no uniformes
- Restricciones de waypoints

El solver lee `reward.csv` directamente, así que cualquier estructura de recompensa funciona si se guarda en ese formato.

---

## 📖 Documentación Adicional

Consulta el código fuente para detalles de implementación:
- **[rl_convnet_simple.c](rl_convnet_simple.c)** - Algoritmo de value iteration
- **[sim/map_generator.py](sim/map_generator.py)** - Generación procedural de mapas
- **[viz/plot_policy.py](viz/plot_policy.py)** - Visualización y simulación
- **[test_pipeline.py](test_pipeline.py)** - Pipeline de automatización

---

**Última actualización**: Febrero 2026  
**Versión**: 3.0 (pipeline unificado con start/goal personalizables y sistema de recompensa híbrido)

