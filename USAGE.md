# Usage Guide - RL Policy Pipeline

Complete guide for generating maps, training policies, and simulating trajectories.

---

## Quick Start

Generate a map, train a policy, and simulate in one command:

```bash
python test_pipeline.py --density 0.4 --seed 42 --start 10,10 --goal 90,90 \
  --tol 1e-3 --k-max 5000 --action all --goal-radius 1
```

**What happens:**
1. Generates map in `sim_maps/map_d0.4_s42_sm3_cr2/`
2. Trains policy using value iteration solver
3. Simulates trajectory and saves plot in map folder

---

## Pipeline Command: `test_pipeline.py`

### Actions

- `--action list`: List all available maps (default)
- `--action train`: Train policy on existing map
- `--action simulate`: Simulate policy on existing map
- `--action all`: Full pipeline (generate + train + simulate)

### Map Generation Parameters

Generate new maps with custom parameters:

```bash
python test_pipeline.py --density 0.3 --seed 123 --start 5,5 --goal 94,94 --action all
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--density` | float | - | Obstacle density (0-1), **required with --seed** |
| `--seed` | int | - | Random seed for reproducibility, **required with --density** |
| `--smooth` | int | 3 | Smoothing passes for obstacle blobs |
| `--carve` | int | 2 | Radius for guaranteed free corridor |
| `--start` | str | "5,5" | Start position as 'x,y' |
| `--goal` | str | "94,94" | Goal position as 'x,y' |

**Map folder naming:** `map_d{density}_s{seed}_sm{smooth}_cr{carve}`

**Example:**
```bash
python test_pipeline.py --density 0.5 --seed 999 --smooth 5 --carve 3 --action all
# Creates: sim_maps/map_d0.5_s999_sm5_cr3/
```

### Solver Parameters

Control training convergence:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--tol` | float | 1e-4 | Convergence tolerance (stop when max_delta < tol) |
| `--k-max` | int | 50000 | Maximum iterations before timeout |

**Examples:**

Fast training (may not fully converge):
```bash
--tol 1e-2 --k-max 1000
```

High-quality training:
```bash
--tol 1e-6 --k-max 100000
```

### Simulation Parameters

Control trajectory generation:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--steps` | int | 10000 | Maximum simulation steps |
| `--action-mode` | str | "argmax" | Action selection: `argmax` (greedy) or `softmax` (stochastic) |
| `--temperature` | float | 1.0 | Temperature for softmax (higher = more exploration) |
| `--goal-radius` | int | 0 | Goal tolerance radius (0 = exact match) |

**Examples:**

Greedy trajectory (deterministic):
```bash
--action-mode argmax --goal-radius 1
```

Exploratory trajectory (stochastic):
```bash
--action-mode softmax --temperature 0.5 --goal-radius 2
```

### Working with Existing Maps

Train on existing map:
```bash
python test_pipeline.py --action train --map map_d0.4_s42_sm3_cr2 --tol 1e-3 --k-max 5000
```

Simulate on existing map:
```bash
python test_pipeline.py --action simulate --map map_d0.4_s42_sm3_cr2 --goal-radius 1
```

Process all maps in batch:
```bash
python test_pipeline.py --action all --all-maps --tol 1e-3 --k-max 1000
```

---

## Manual Map Generation: `sim/map_generator.py`

Generate maps without training:

```bash
python sim/map_generator.py --density 0.35 --seed 42 --smooth 3 --carve 2 \
  --start 10,10 --goal 85,85 --goal-reward 10.0 --goal-radius 2
```

**Advanced Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--width` | int | 100 | Map width |
| `--height` | int | 100 | Map height |
| `--obstacle-penalty` | float | -1.0 | Reward for obstacle cells |
| `--free-reward` | float | 0.0 | Base reward for free cells |
| `--reward-min` | float | 0.0 | Minimum reward for free cells far from obstacles |
| `--reward-max` | float | 1.0 | Maximum reward for free cells far from obstacles |
| `--goal-reward` | float | 5.0 | High reward at/near goal (attracts policy) |
| `--goal-radius` | int | 1 | Radius around goal with reward boost |
| `--out` | str | "sim_maps" | Output directory |

**How reward works:**

1. **Obstacle cells:** Fixed penalty (`--obstacle-penalty = -1.0`)
2. **Free cells:** Linear interpolation based on distance to obstacles:
   - Far from obstacles → `--reward-max`
   - Near obstacles → `--reward-min`
3. **Goal region:** Strong reward (`--goal-reward = 5.0`) within `--goal-radius` of goal

This creates a **safe path preference** (avoids obstacles) while **guaranteeing convergence to goal**.

**Output files in map folder:**
- `obstacles.csv`: Binary grid (0=free, 1=obstacle)
- `obstacles.png`: Visualization with start (green) and goal (red star)
- `reward.csv`: Float reward per cell
- `start_goal.csv`: Single line with `sx,sy,gx,gy`
- `policy.txt`: Generated after training
- `policy_path.png`: Generated after simulation

---

## Manual Training: `rl_convnet_simple.exe`

Compile and run solver directly:

```bash
gcc -O2 rl_convnet_simple.c -o rl_convnet_simple.exe -lm

./rl_convnet_simple.exe --reward sim_maps/map_d0.4_s42_sm3_cr2/reward.csv \
  --tol 1e-3 --k-max 5000
```

**Key features:**
- 3×3 spatial convolution kernels per orientation
- Adaptive convergence detection
- Saves policy to both root and map folder

**Output messages:**
```
[t=500] max_delta=0.012345
Converged at iteration 500 with tol=0.001
Policy saved to: policy.txt and sim_maps/map_d0.4_s42_sm3_cr2/policy.txt
```

---

## Manual Visualization: `viz/plot_policy.py`

### Simulate Trajectory

```bash
python viz/plot_policy.py --simulate \
  --reward sim_maps/map_d0.4_s42_sm3_cr2/reward.csv \
  --action-mode argmax --goal-radius 1 \
  --out-dir sim_maps/map_d0.4_s42_sm3_cr2
```

**Auto-detection:** Automatically finds `policy.txt` in map folder (no `--input` needed)

### Plot Policy Grids

Single orientation argmax:
```bash
python viz/plot_policy.py --argmax --orient 0 --arrows --stride 5
```

All Q-values for orientation:
```bash
python viz/plot_policy.py --all-actions --orient 0
```

Mosaic of all orientations:
```bash
python viz/plot_policy.py --mosaic --arrows --stride 3
```

All grids individually:
```bash
python viz/plot_policy.py --all
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--input` | Path to policy.txt (auto-detected if omitted) |
| `--reward` | Path to reward.csv for simulation |
| `--orient` | Orientation index (0-7) |
| `--action` | Action index (0-5) |
| `--arrows` | Overlay direction arrows |
| `--stride` | Arrow sampling stride |
| `--start-x`, `--start-y` | Simulation start (defaults from start_goal.csv) |
| `--start-orient` | Initial orientation |
| `--steps` | Max simulation steps |
| `--action-mode` | `argmax` (greedy) or `softmax` (stochastic) |
| `--temperature` | Softmax temperature |
| `--goal-radius` | Goal tolerance radius |
| `--out-dir` | Output directory |
| `--show` | Display plot window |
| `--print-table` | Print action direction table |

---

## Complete Workflow Examples

### Example 1: High-Density Map with Safe Navigation

```bash
python test_pipeline.py \
  --density 0.6 --seed 777 \
  --start 5,5 --goal 95,95 \
  --smooth 5 --carve 3 \
  --tol 1e-4 --k-max 20000 \
  --goal-radius 2 \
  --action all
```

**What this does:**
- Generates dense obstacle map (60%)
- Wide safe corridors (carve=3, smooth=5)
- High-quality training (tol=1e-4)
- Tolerant goal region (radius=2)

### Example 2: Quick Testing

```bash
python test_pipeline.py \
  --density 0.3 --seed 123 \
  --tol 1e-2 --k-max 500 \
  --goal-radius 1 \
  --action all
```

**What this does:**
- Simple map (30% obstacles)
- Fast training (low iterations)
- Quick validation workflow

### Example 3: Exploratory Trajectories

```bash
python test_pipeline.py \
  --density 0.4 --seed 42 \
  --start 10,10 --goal 80,80 \
  --tol 1e-3 --k-max 10000 \
  --action-mode softmax --temperature 0.3 \
  --goal-radius 1 \
  --action all
```

**What this does:**
- Stochastic action selection (softmax)
- Low temperature = mostly optimal with small exploration
- Produces varied trajectories on repeated runs

### Example 4: Batch Processing

Generate multiple maps with different seeds:

```bash
for seed in 100 200 300 400 500; do
  python test_pipeline.py \
    --density 0.4 --seed $seed \
    --tol 1e-3 --k-max 5000 \
    --goal-radius 1 \
    --action all
done
```

Process all existing maps:

```bash
python test_pipeline.py --action all --all-maps --tol 1e-3 --k-max 5000 --goal-radius 1
```

---

## Troubleshooting

### "Simulation stop reason: cycle"

**Cause:** Agent revisited same state (position + orientation) before reaching goal.

**Solutions:**
1. Increase `--goal-radius` (e.g., from 0 to 1 or 2)
2. Use `--action-mode softmax --temperature 0.5` to break cycles
3. Increase `--goal-reward` in map generation (e.g., from 5.0 to 10.0)
4. Reduce `--tol` or increase `--k-max` for better convergence

### "Simulation stop reason: hit-obstacle"

**Cause:** Policy quality insufficient or map too difficult.

**Solutions:**
1. Decrease `--tol` (e.g., from 1e-2 to 1e-4)
2. Increase `--k-max` (e.g., from 1000 to 10000)
3. Increase `--carve` radius for wider corridors
4. Increase `--smooth` passes for cleaner obstacles
5. Lower `--density` for fewer obstacles

### "Warning: hit k-max without reaching tol"

**Cause:** Training iterations exhausted before convergence.

**Solutions:**
1. Increase `--k-max` (e.g., double it)
2. Increase `--tol` if precision isn't critical
3. Check map difficulty (very high density may be unsolvable)

### Policy doesn't reach goal

**Cause:** Reward doesn't sufficiently attract to goal.

**Solutions:**
1. Increase `--goal-reward` when generating map (e.g., 10.0 or 20.0)
2. Increase `--goal-radius` to create larger attractive region
3. Ensure map generation includes goal reward (automatic in new version)

---

## File Structure

```
RISCVsummit/
├── test_pipeline.py          # Main automation script
├── rl_convnet_simple.c        # C solver (value iteration)
├── rl_convnet_simple.exe      # Compiled solver
├── sim/
│   ├── map_generator.py       # Procedural map generation
│   └── map_visualize.py       # (deprecated, use plot_policy.py)
├── viz/
│   └── plot_policy.py         # Visualization and simulation
├── sim_maps/
│   └── map_d{d}_s{s}_sm{sm}_cr{cr}/  # Generated maps
│       ├── obstacles.csv
│       ├── obstacles.png
│       ├── reward.csv
│       ├── start_goal.csv
│       ├── policy.txt         # After training
│       └── policy_path.png    # After simulation
└── policy_plots/              # Legacy output folder
```

---

## Tips for Good Results

1. **Start simple:** Use low density (0.2-0.4) and quick training (k-max=1000) for testing
2. **Goal region:** Always use `--goal-radius 1` or higher to avoid cycle issues
3. **Safe paths:** Default `--goal-reward 5.0` balances safety and goal attraction
4. **Convergence:** For publication-quality results, use `--tol 1e-5` and `--k-max 50000`
5. **Reproducibility:** Always specify `--seed` for deterministic results
6. **Visualization:** Use `--arrows --stride 5` to see policy direction fields

---

## Advanced: Custom Reward Functions

Edit `build_reward_map()` in [sim/map_generator.py](sim/map_generator.py) to implement:

- Distance-based rewards to goal (instead of obstacles)
- Multi-goal scenarios
- Non-uniform terrain costs
- Waypoint constraints

The solver reads `reward.csv` directly, so any reward structure works if saved in that format.

---

## Performance Notes

- **Map generation:** < 1 second
- **Training:** 1-60 seconds depending on k-max and convergence
- **Simulation:** < 1 second
- **Full pipeline:** Typically 5-30 seconds

**Memory usage:**
- Policy file: ~6 MB for 100×100 grid with 8 orientations and 6 actions
- All files per map: ~12-15 MB

---

## Citation

If you use this code, please reference the project appropriately.

```bibtex
@misc{rlconvnet2026,
  title={RL Convnet Policy: Value Iteration with Spatial Kernels},
  author={Your Name},
  year={2026},
  url={https://github.com/yourrepo/RISCVsummit}
}
```
