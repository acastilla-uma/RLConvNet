# README_TESTING: Guía Completa de Pruebas

Documentación detallada para usar el framework RL ConvNet con todos los parámetros y opciones.

---

## Table of Contents

1. [Generar Mapas](#generar-mapas)
2. [Entrenar Políticas](#entrenar-políticas)
3. [Simular Políticas](#simular-políticas)
4. [Test Pipeline (Automatización)](#test-pipeline)
5. [Ejemplos Prácticos](#ejemplos-prácticos)
6. [Troubleshooting](#troubleshooting)

---

## Generar Mapas

### Uso Básico

```bash
python sim/map_generator.py --density 0.35 --seed 42
```

Resultado:
```
✓ Generated map in: sim_maps\map_d0.35_s42_sm3_cr2
  - obstacles.csv (0 free, 1 obstacle)
  - reward.csv (float reward per cell)
  - start_goal.csv (sx,sy,gx,gy)
  - obstacles.png (visualization)
```

### Parámetros Completos

```bash
python sim/map_generator.py \
    --density <float>            # Densidad de obstáculos (0.0-1.0, default: 0.35)
    --seed <int>                 # Seed aleatorio (default: 42)
    --smooth <int>               # Pasadas de suavizado (default: 3)
    --carve <int>                # Radio de carving para garantizar path (default: 2)
    --width <int>                # Ancho del grid (default: 100)
    --height <int>               # Alto del grid (default: 100)
    --start <str>                # Posición inicio "x,y" (default: "5,5")
    --goal <str>                 # Posición meta "x,y" (default: "94,94")
    --out <str>                  # Carpeta base (default: "sim_maps")
```

### Ejemplos

```bash
# Mapa fácil (baja densidad)
python sim/map_generator.py --density 0.1 --seed 1 --smooth 1 --carve 1

# Mapa estándar
python sim/map_generator.py --density 0.35 --seed 42 --smooth 3 --carve 2

# Mapa difícil (alta densidad)
python sim/map_generator.py --density 0.7 --seed 100 --smooth 5 --carve 3

# Generar suite de mapas
for d in 0.2 0.4 0.6; do
    python sim/map_generator.py --density $d --seed 42
done
```

### Nomenclatura de Carpetas

Formato automático: `map_d{D}_s{S}_sm{SM}_cr{CR}`

- `d{D}`: Densidad (ej: d0.35)
- `s{S}`: Seed (ej: s42)
- `sm{SM}`: Suavizado (ej: sm3)
- `cr{CR}`: Carve radius (ej: cr2)

**Ejemplo**: `map_d0.35_s42_sm3_cr2` = 35% densidad, seed 42, 3 pasadas smooth, radio 2

---

## Entrenar Políticas

### Compilación (Una sola vez)

```bash
# Compilación estándar
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm

# Compilación con warnings (debugging)
gcc -Wall -Wextra rl_convnet_simple.c -o rl_convnet_simple.exe -lm
```

### Uso Básico

```bash
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv
```

Salida:
```
[t=100] max_delta=1.83198
[t=200] max_delta=1.76438
Converged at iteration 128 (delta=1.79999)
RLConvNet simple demo. V(goal)=165.842428 iters=128 (converged) tol=1.8
```

### Parámetros Completos

```bash
./rl_convnet_simple.exe \
    --reward <path>              # CSV de recompensas (requerido)
    --tol <float>                # Tolerancia de convergencia (default: 1e-4)
    --k-max <int>                # Máximas iteraciones (default: 1000)
    --n-orient <int>             # Orientaciones (1-8, default: 8)
    --n-actions <int>            # Acciones per orientación (1-6, default: 6)
    --allowed-actions <int>      # Solo primeras N acciones (default: 3)
    --help                       # Mostrar ayuda
```

### Ejemplos de Entrenamiento

```bash
# Convergencia rápida (laxa)
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv \
    --tol 1e-2 --k-max 5000

# Convergencia precisa (estricta)
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv \
    --tol 1e-6 --k-max 100000

# Arquitectura reducida (4 orientaciones, 2 acciones)
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv \
    --n-orient 4 --n-actions 2 --allowed-actions 1

# Máxima exploración
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv \
    --n-orient 8 --n-actions 6 --allowed-actions 6
```

### Interpretación de Salida

```
[t=100] max_delta=1.83198      ← Cada 100 iters: delta máximo
[t=200] max_delta=1.76438      ← Converge cuando delta < tol
Converged at iteration 128     ← 128 iteraciones necesarias
(delta=1.79999)                ← Delta final
V(goal)=165.842428             ← Valor en la meta
iters=128 (converged)          ← Status: converged OR maxed
tol=1.8                        ← Tolerancia usada
```

---

## Simular Políticas

### Uso Básico

```bash
python viz/plot_policy.py --simulate --action-mode argmax
```

### Parámetros Completos

```bash
python viz/plot_policy.py \
    --simulate                          # Activar simulación
    --action-mode {argmax|softmax}      # Selección acción (default: argmax)
    --temperature <float>               # Temp. softmax (default: 1.0)
    --steps <int>                       # Max pasos simulación (default: 10000)
    --seed-sim <int>                    # Seed aleatorio simulación
    --reward <path>                     # Path a reward.csv (si está fuera de cwd)
    --all-actions                       # Mostrar Q-values todas acciones
    --orient <int>                      # Orientación específica (0-7)
```

### Ejemplos

```bash
# Simulación determinista (greedy)
python viz/plot_policy.py --simulate --action-mode argmax --steps 5000

# Exploración suave (T < 1)
python viz/plot_policy.py --simulate --action-mode softmax --temperature 0.5 --steps 5000

# Exploración estándar (T = 1)
python viz/plot_policy.py --simulate --action-mode softmax --temperature 1.0 --steps 5000

# Exploración agresiva (T > 1)
python viz/plot_policy.py --simulate --action-mode softmax --temperature 2.0 --steps 5000

# Mostrar todas las acciones
python viz/plot_policy.py --all-actions

# Visualizar orientación específica
python viz/plot_policy.py --all-actions --orient 0
```

---

## Test Pipeline

Script automatizado para train + simulate en batch.

### Listar Mapas

```bash
python test_pipeline.py --action list
```

Salida:
```
Available maps:
  map_d0.1_s55_sm1_cr1
    Files: obstacles.csv, obstacles.png, reward.csv, start_goal.csv
  map_d0.35_s42_sm3_cr2
    Files: obstacles.csv, obstacles.png, reward.csv, start_goal.csv
  map_d0.5_s100_sm4_cr1
    Files: obstacles.csv, obstacles.png, reward.csv, start_goal.csv
```

### Entrenar Mapa Específico

```bash
# Entrenar con parámetros default
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2

# Entrenar con parámetros personalizados
python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2 \
    --tol 1e-3 --k-max 50000
```

### Simular

```bash
# Determinista
python test_pipeline.py --action simulate --map map_d0.35_s42_sm3_cr2 \
    --action-mode argmax

# Estocástico
python test_pipeline.py --action simulate --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax --temperature 1.5 --steps 5000
```

### Train + Simulate (Flujo Completo)

```bash
# Un mapa
python test_pipeline.py --action all --map map_d0.35_s42_sm3_cr2

# Todos los mapas
python test_pipeline.py --action all --all-maps
```

---

## Ejemplos Prácticos

### Ejemplo 1: Generar y Probar Suite de Densidades

```bash
# Generar mapas con diferentes densidades (mismo seed)
for density in 0.1 0.2 0.3 0.4 0.5 0.6; do
    python sim/map_generator.py --density $density --seed 42
done

# Entrenar todos
python test_pipeline.py --action all --all-maps

# Resultado: 6 carpetas en sim_maps/, todas entrenadas y simuladas
```

### Ejemplo 2: Estudiar Efecto de Convergencia

```bash
map="map_d0.35_s42_sm3_cr2"

# Tolerancia laxa (converge rápido)
echo "Tolerancia laxa (tol=1e-2):"
./rl_convnet_simple.exe --reward sim_maps/$map/reward.csv --tol 1e-2 --k-max 10000

# Tolerancia intermedia
echo "Tolerancia intermedia (tol=1e-4):"
./rl_convnet_simple.exe --reward sim_maps/$map/reward.csv --tol 1e-4 --k-max 50000

# Tolerancia estricta (converge lentamente)
echo "Tolerancia estricta (tol=1e-6):"
./rl_convnet_simple.exe --reward sim_maps/$map/reward.csv --tol 1e-6 --k-max 100000

# Comparar resultados: número de iteraciones
# Esperado: más iteraciones con tol más estricta
```

### Ejemplo 3: Análisis de Exploración vs Explotación

```bash
map="map_d0.35_s42_sm3_cr2"

# Primero entrenar
./rl_convnet_simple.exe --reward sim_maps/$map/reward.csv --tol 1e-3

# Luego simular con diferentes estrategias
echo "Greedy (determinista):"
python viz/plot_policy.py --simulate --action-mode argmax --steps 1000

echo "Softmax T=0.3 (muy greedy):"
python viz/plot_policy.py --simulate --action-mode softmax --temperature 0.3 --steps 1000

echo "Softmax T=1.0 (balanceado):"
python viz/plot_policy.py --simulate --action-mode softmax --temperature 1.0 --steps 1000

echo "Softmax T=3.0 (muy exploratorio):"
python viz/plot_policy.py --simulate --action-mode softmax --temperature 3.0 --steps 1000

# Observar: longitud de path, ciclos detectados, etc.
```

### Ejemplo 4: Visualizar Mapas

```bash
# Visualizar obstáculos
python sim/map_visualize.py --folder sim_maps/map_d0.35_s42_sm3_cr2

# Visualizar con mapa de recompensas
python sim/map_visualize.py --folder sim_maps/map_d0.35_s42_sm3_cr2 --show-reward

# Guardar a archivo
python sim/map_visualize.py --folder sim_maps/map_d0.35_s42_sm3_cr2 \
    --save my_map.png
```

---

## Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| "reward.csv not found" | Mapa no existe | Generar: `python sim/map_generator.py` |
| "Solver tarda >1 min" | Tolerancia muy estricta | Aumentar `--tol` (ej: 1e-2) |
| "Converged at k-max" | No alcanzó tolerancia | Aumentar `--k-max` o relajar `--tol` |
| Simulación se cicla infinito | Normal en mapas complejos | Usar `--steps 1000` para limitar |
| Mapa muy vacío | Densidad demasiado baja | Aumentar `--density` (0.3-0.5 típico) |
| Mapa muy lleno | Densidad demasiado alta | Disminuir `--density` |
| gcc no encontrado | Compilador no instalado | Instalar MinGW o GCC nativo |
| import matplotlib failed | librería faltante | `pip install matplotlib` |
| "invalid folder" message | Carpeta no existe | Usar `test_pipeline.py --action list` |

---

## Formato de Datos

### obstacles.csv
- Grid 100×100
- Valores: 0 (libre) o 1 (obstáculo)
- Separador: comas
- Ejemplo primer row: `1,1,1,0,0,1,...`

### reward.csv
- Grid 100×100
- Valores: floats
- Obstáculos típicamente -1.0
- Células libres: 0.0-1.0 (gradiente distancia)
- Ejemplo: `0.0,0.1,0.2,0.5,1.0,...`

### start_goal.csv
- Una sola línea
- Formato: `sx,sy,gx,gy`
- Ejemplo: `5,5,94,94`
- (sy, gy) son coordenadas Y (rows), (sx, gx) son X (cols)

### policy.txt
- Encabezado con metadatos
- Grids de Q-values para cada (orientación, acción)
- Legible por plot_policy.py automáticamente

---

## Parámetros Recomendados

### Para Mapas Fáciles (densidad < 0.3)
```bash
--tol 1e-2      # Convergencia rápida
--k-max 5000    # Iteraciones limitadas
--n-orient 4    # Pocas orientaciones
--n-actions 2   # Pocas acciones
```

### Para Mapas Estándar (densidad 0.3-0.5)
```bash
--tol 1e-4      # Convergencia moderada
--k-max 50000   # Iteraciones razonables
--n-orient 8    # Todas orientaciones
--n-actions 6   # Todas acciones (solo 3 permitidas)
```

### Para Mapas Difíciles (densidad > 0.5)
```bash
--tol 1e-6      # Convergencia precisa
--k-max 100000  # Muchas iteraciones
--n-orient 8    # Todas orientaciones
--n-actions 6   # Todas acciones
```

---

## Performance Tips

1. **Compilación**: usar `-O2` (optimization level 2)
2. **Tolerancia**: aumentar si solo necesitas policy aproximada
3. **Mapas**: densidades 0.2-0.5 convergen más rápido
4. **Batch**: usar `test_pipeline.py --all-maps` para procesar varios

---

**Última actualización**: 12 Feb 2026
