# GUÍA RÁPIDA: Primeros Pasos

**Comenzar en 2 minutos** ⚡

---

## 1️⃣ Compilar (30 segundos)

```bash
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm
```

---

## 2️⃣ Generar Mapa (20 segundos)

```bash
python sim/map_generator.py --density 0.35
```

✓ Crea: `sim_maps/map_d0.35_s42_sm3_cr2/`

---

## 3️⃣ Entrenar (10-30 segundos)

```bash
./rl_convnet_simple.exe --reward sim_maps/map_d0.35_s42_sm3_cr2/reward.csv
```

Salida: `Converged at iteration 128 (delta=1.79999)`

---

## 4️⃣ Simular (5 segundos)

```bash
python viz/plot_policy.py --simulate
```

✓ Genera: `policy_plots/policy_path.png`

---

## ✅ ¡LISTO!

El agente entrenado navega del inicio a la meta evitando obstáculos.

---

## 📚 Documentación Disponible

| Archivo | Para qué |
|---------|----------|
| **README.md** | Intro rápida, estructura del proyecto |
| **README_TESTING.md** | Todos los parámetros, ejemplos completos |
| **RESTRUCTURING_SUMMARY.md** | Cambios recientes, estructura de carpetas |
| **Este archivo** | Primeros pasos (2 minutos) |

---

## 🎮 Comandos Más Útiles

```bash
# Ver help
./rl_convnet_simple.exe --help
python test_pipeline.py --help

# Listar mapas
python test_pipeline.py --action list

# Train + simulate todo
python test_pipeline.py --action all --all-maps

# Simulación con exploración
python viz/plot_policy.py --simulate --action-mode softmax --temperature 1.5
```

---

**Más detalles**: consulta [README.md](README.md)
