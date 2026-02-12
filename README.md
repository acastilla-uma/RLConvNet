# RL Convnet Testing Framework

Nueva estructura reorganizada para facilitar pruebas del pipeline de aprendizaje por refuerzo.

## Estructura de Carpetas

```
RISCVsummit/
├── rl_convnet_simple.exe         # Solver entrenador (ejecutable compilado)
├── test_pipeline.py              # Script de utilidad para pruebas
├── sim/
│   ├── map_generator.py          # Generador de mapas (crea carpetas con nombres descriptivos)
│   └── map_visualize.py          # Visualizador de mapas
├── sim_maps/                     # Base de datos de mapas generados
│   ├── map_d0.35_s42_sm3_cr2/   # Nombre = parámetros (density, seed, smooth, carve)
│   │   ├── obstacles.csv         # Grid de obstáculos (0=libre, 1=obstáculo)
│   │   ├── reward.csv            # Grid de recompensas
│   │   ├── start_goal.csv        # Posiciones de inicio y meta
│   │   └── obstacles.png         # Visualización automática
│   ├── map_d0.5_s100_sm4_cr1/
│   │   ├── obstacles.csv
│   │   ├── reward.csv
│   │   ├── start_goal.csv
│   │   └── obstacles.png
│   └── map_d0.2_s999_sm2_cr3/
│       └── ...
├── policy.txt                    # Política entrenada (generada por solver)
└── policy_plots/                 # Visualizaciones de políticas
```

## Flujo de Trabajo

### 1. Generar Mapas

El generador de mapas crea automáticamente carpetas con nombres legibles basados en los parámetros:

```bash
# Generar un mapa con density=0.35, seed=42, smooth=3, carve_radius=2
python sim/map_generator.py --density 0.35 --seed 42 --smooth 3 --carve 2

# Resultado:
# ✓ Generated map in: sim_maps\map_d0.35_s42_sm3_cr2
#   - obstacles.csv
#   - reward.csv
#   - start_goal.csv
#   - obstacles.png  (automáticamente generado)
```

**Parámetros disponibles:**
- `--density 0.0-1.0`: Densidad de obstáculos (default: 0.35)
- `--seed INT`: Seed para reproducibilidad (default: 42)
- `--smooth INT`: Pasadas de suavizado (default: 3)
- `--carve INT`: Radio de depuración para garantizar camino (default: 2)
- `--width INT`: Ancho del mapa (default: 100)
- `--height INT`: Alto del mapa (default: 100)

### 2. Visualizar Mapas

```bash
# Visualizar un mapa específico
python sim/map_visualize.py --folder sim_maps/map_d0.35_s42_sm3_cr2

# Visualizar con mapa de recompensas
python sim/map_visualize.py --folder sim_maps/map_d0.35_s42_sm3_cr2 --show-reward

# Listar mapas disponibles (se muestra automáticamente si carpeta no existe)
python sim/map_visualize.py --folder invalid/path
```

### 3. Entrenar Políticas (Solver)

```bash
# Compilar (una sola vez)
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm

# Entrenar con un mapa específico
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv

# Con parámetros personalizados
./rl_convnet_simple.exe \
    --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv \
    --tol 1e-4 \
    --k-max 50000 \
    --n-orient 8 \
    --n-actions 6 \
    --allowed-actions 3
```

**Parámetros del solver:**
- `--reward PATH`: Ruta a reward.csv (requerido)
- `--tol FLOAT`: Tolerancia de convergencia (default: 1e-4)
- `--k-max INT`: Máximo de iteraciones (default: 1000)
- `--n-orient INT`: Número de orientaciones (default: 8)
- `--n-actions INT`: Acciones por orientación (default: 6)
- `--allowed-actions INT`: Solo usar primeras N acciones (default: 3)

### 4. Usar el Script de Utilidad (test_pipeline.py)

Script automatizado que maneja el flujo completo de train -> simulate:

```bash
# Listar todos los mapas disponibles
python test_pipeline.py --action list

# Entrenar política para un mapa específico
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2

# Simular política
python test_pipeline.py --action simulate --map map_d0.35_s42_sm3_cr2 --action-mode argmax

# Entrenar Y simular (flujo completo)
python test_pipeline.py --action all --map map_d0.35_s42_sm3_cr2

# Entrenar con parámetros personalizados
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2 --tol 1e-3 --k-max 100000

# Procesar todos los mapas
python test_pipeline.py --action all --all-maps

# Simular con softmax (exploración)
python test_pipeline.py \
    --action simulate \
    --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax \
    --temperature 1.5 \
    --steps 5000
```

**Opciones del script:**
- `--action {list|train|simulate|all}`: Acción a ejecutar
- `--map NOMBRE`: Mapa específico (sin argumentos lista todos)
- `--all-maps`: Procesar todos los mapas disponibles
- `--tol FLOAT`: Tolerancia del solver
- `--k-max INT`: Máximas iteraciones del solver
- `--steps INT`: Máximos pasos de simulación (default: 10000)
- `--action-mode {argmax|softmax}`: Modo de selección (default: argmax)
- `--temperature FLOAT`: Temperatura para softmax (default: 1.0)

## Ejemplos Prácticos

### Ejemplo 1: Generar y probar múltiples mapas

```bash
# Generar mapas con diferentes densidades
python sim/map_generator.py --density 0.2 --seed 1
python sim/map_generator.py --density 0.4 --seed 1
python sim/map_generator.py --density 0.6 --seed 1

# Entrenar y simular todo en una línea
python test_pipeline.py --action all --all-maps
```

### Ejemplo 2: Estudiar efecto de convergencia

```bash
# Mapa fácil con tolerancia laxa (converge rápido)
python test_pipeline.py --action train --map map_d0.2_s999_sm2_cr3 --tol 2.0 --k-max 5000

# Mismo mapa con tolerancia estricta (converge lentamente)
python test_pipeline.py --action train --map map_d0.2_s999_sm2_cr3 --tol 1e-5 --k-max 100000

# Comparar tiempos de convergencia en la salida del solver
# [t=100] max_delta=...
# Converged at iteration X
```

### Ejemplo 3: Análisis de comportamiento de exploración

```bash
# Simular con política determinista (argmax)
python test_pipeline.py \
    --action simulate \
    --map map_d0.35_s42_sm3_cr2 \
    --action-mode argmax

# Simular con exploración suave
python test_pipeline.py \
    --action simulate \
    --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax \
    --temperature 0.5

# Simular con exploración alta
python test_pipeline.py \
    --action simulate \
    --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax \
    --temperature 2.0
```

## Nomenclatura de Carpetas de Mapas

Formato: `map_dX_sY_smZ_crW`

- `d`: Densidad (ej: 0.35)
- `s`: Seed (ej: 42)
- `sm`: Suavizado (ej: 3)
- `cr`: Carve radius (ej: 2)

Ejemplos:
- `map_d0.35_s42_sm3_cr2` → 35% densidad, seed 42, 3 pasadas suavizado, radio 2
- `map_d0.5_s100_sm4_cr1` → 50% densidad, seed 100, 4 pasadas suavizado, radio 1

## Beneficios de la Nueva Estructura

✅ **Organización clara**: Cada mapa en su propia carpeta
✅ **Nombres descriptivos**: Los parámetros están en el nombre de la carpeta
✅ **Visualización automática**: obstacles.png generado automáticamente
✅ **Reproducibilidad**: Seeds en el nombre facilitan reproducir experimentos
✅ **Pruebas facilitadas**: Script test_pipeline.py automatiza train+simulate
✅ **Escalabilidad**: Fácil mantener múltiples configuraciones

## Próximos Pasos

1. Generar una batería de mapas con diferentes densidades y seeds
2. Entrenar políticas con diferentes tolerancias de convergencia
3. Comparar tiempos de entrenamiento
4. Analizar calidad de políticas con argmax vs softmax
5. Visualizar y documentar resultados
