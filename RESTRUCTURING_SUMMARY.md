# Reestructuración Completada: Facilitar Pruebas con Mapas Organizados

## Resumen de Cambios

Se ha reestructurado completamente el sistema de generación y prueba de mapas para facilitar experimentos reproducibles.

### ✅ Cambios Realizados

#### 1. **map_generator.py** - Nombres descriptivos de carpetas
- **Antes**: Todos los archivos en `sim_maps/` (desordenado)
- **Ahora**: Cada mapa en su propia carpeta con nombre descriptivo

```
Formato: sim_maps/map_d{density}_s{seed}_sm{smooth}_cr{carve}/
Ejemplos:
  map_d0.35_s42_sm3_cr2/    → 35% densidad, seed 42, 3 smooth, radio 2
  map_d0.5_s100_sm4_cr1/    → 50% densidad, seed 100, 4 smooth, radio 1
  map_d0.1_s55_sm1_cr1/     → 10% densidad, seed 55, 1 smooth, radio 1
```

#### 2. **Visualización automática**
- El generador ahora crea automáticamente `obstacles.png` en cada carpeta
- Muestra obstáculos (negro), espacio libre (blanco), inicio (verde) y meta (roja)
- Sin necesidad de ejecutar `map_visualize.py` manualmente

#### 3. **map_visualize.py** - Mejor navegación
- Ahora apunta por defecto a la primera carpeta de mapa
- Muestra lista de mapas disponibles si se proporciona carpeta inválida
- Soporta `--show-reward` para visualizar mapa de recompensas
- Mejor manejo de errores

#### 4. **test_pipeline.py** - Script de utilidad
- Nueva herramienta para automatizar: generar → entrenar → simular
- Funciones principales:
  - `--action list`: Lista todos los mapas disponibles
  - `--action train`: Entrenar política en mapa específico
  - `--action simulate`: Simular política entrenada
  - `--action all`: Flujo completo (train + simulate)
  - `--all-maps`: Procesar todos los mapas

### 📊 Estructura Actual

```
sim_maps/
├── map_d0.1_s55_sm1_cr1/          ← Mapa fácil (10% densidad)
│   ├── obstacles.csv              ← Grid 0/1 (0=libre, 1=obstáculo)
│   ├── obstacle.png               ← 🆕 Visualización automática
│   ├── reward.csv                 ← Gradiente de distancia
│   └── start_goal.csv             ← Posiciones inicio/meta
│
├── map_d0.2_s999_sm2_cr3/         ← Mapa moderado (20% densidad)
│   ├── obstacles.csv
│   ├── obstacles.png
│   ├── reward.csv
│   └── start_goal.csv
│
├── map_d0.35_s42_sm3_cr2/         ← Mapa estándar (35% densidad)
│   ├── obstacles.csv
│   ├── obstacles.png
│   ├── reward.csv
│   └── start_goal.csv
│
└── map_d0.5_s100_sm4_cr1/         ← Mapa difícil (50% densidad)
    ├── obstacles.csv
    ├── obstacles.png
    ├── reward.csv
    └── start_goal.csv
```

### 🔄 Flujo de Trabajo Típico

```bash
# 1. Generar mapas con diferentes parámetros
python sim/map_generator.py --density 0.3 --seed 1
python sim/map_generator.py --density 0.4 --seed 42
python sim/map_generator.py --density 0.6 --seed 100

# 2. Entrenar y simular automáticamente
python test_pipeline.py --action all --all-maps

# 3. Entrenar un mapa específico con parámetros personalizados
python test_pipeline.py \
    --action train \
    --map map_d0.35_s42_sm3_cr2 \
    --tol 1e-3 \
    --k-max 100000

# 4. Simular con exploración
python test_pipeline.py \
    --action simulate \
    --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax \
    --temperature 1.5
```

### 📈 Beneficios

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Organización** | Archivos sueltos en sim_maps/ | Carpetas ordenadas por parámetros |
| **Reproducibilidad** | Difícil recordar qué parámetros | Nombre de carpeta contiene todos |
| **Visualización** | Manual (ejecutar map_visualize.py) | Automática (obstacles.png) |
| **Pruebas** | Error-prone (múltiples comandos) | Automatizado (test_pipeline.py) |
| **Escalabilidad** | Confuso con muchos mapas | Fácil mantener docenas de mapas |

### 📝 Ejemplos de Uso

#### Generar batería de mapas para experimentos

```bash
# Experimento: Efecto de densidad
for density in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    python sim/map_generator.py --density $density --seed 42
done

# Resultado: 7 mapas en carpetas organizadas
# map_d0.1_s42_sm3_cr2/, map_d0.2_s42_sm3_cr2/, ...
```

#### Entrenar en mapas fáciles primero

```bash
# Mapas de baja densidad para debug
python test_pipeline.py --action train --map map_d0.1_s55_sm1_cr1 --tol 1e-2

# Mapas progresivamente más difíciles
python test_pipeline.py --action all --all-maps
```

#### Comparar estrategias de exploración

```bash
# Determinista (mejor política)
python test_pipeline.py --action simulate --map map_d0.35_s42_sm3_cr2 \
    --action-mode argmax --steps 5000

# Exploración suave
python test_pipeline.py --action simulate --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax --temperature 0.5 --steps 5000

# Exploración agresiva
python test_pipeline.py --action simulate --map map_d0.35_s42_sm3_cr2 \
    --action-mode softmax --temperature 2.0 --steps 5000
```

### 🚀 Próximos Pasos Sugeridos

1. **Generar batería completa de mapas**
   ```bash
   python sim/map_generator.py --density 0.2 --seed 100
   python sim/map_generator.py --density 0.4 --seed 200
   python sim/map_generator.py --density 0.6 --seed 300
   ```

2. **Entrenar y medir convergencia**
   ```bash
   for tol in 1e-2 1e-3 1e-4 1e-5; do
       python test_pipeline.py --action train --map map_d0.35_s42_sm3_cr2 --tol $tol
       # Registrar número de iteraciones
   done
   ```

3. **Análisis de comportamiento**
   - Comparar argmax vs softmax
   - Medir tasas de ciclos/éxito
   - Graficar caminos en diferentes estrategias

### 📖 Documentación

Ver [README_TESTING.md](README_TESTING.md) para documentación completa con:
- Parámetros detallados de cada herramienta
- Todas las opciones de CLI
- Más ejemplos prácticos
- Guía de troubleshooting

### ✨ Características Destacadas

✅ **Reproducibilidad garantizada**: parámetros en nombre de carpeta
✅ **Visualización inmediata**: PNG generado automáticamente
✅ **Escalabilidad**: fácil mantener docenas de configuraciones
✅ **Automatización**: test_pipeline.py para flujos complejos
✅ **Organización clara**: estructura visual fácil de navegar
✅ **Flexibilidad**: soporta múltiples estrategias de entrenamiento/simulación
