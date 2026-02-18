import argparse
import os
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as exc:
    missing = exc.name if hasattr(exc, "name") else str(exc)
    print(f"Missing dependency: {missing}. Install with: pip install numpy matplotlib")
    sys.exit(1)


def load_policy(path):
    policy = {}
    current_key = None
    current_rows = []

    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                if current_key is not None and current_rows:
                    policy[current_key] = np.array(current_rows, dtype=float)
                    current_key = None
                    current_rows = []
                continue
            if line.startswith("#"):
                if current_key is not None and current_rows:
                    policy[current_key] = np.array(current_rows, dtype=float)
                    current_rows = []
                parts = line.replace("#", "").strip().split()
                orient = int(parts[0].split("=")[1])
                action = int(parts[1].split("=")[1])
                current_key = (orient, action)
                continue
            row = [float(v.replace("-nan(ind)", "nan")) for v in line.split(",")]
            current_rows.append(row)

    if current_key is not None and current_rows:
        policy[current_key] = np.array(current_rows, dtype=float)

    return policy


def find_policy_path(reward_path=None):
    """Find policy.txt in multiple locations with priority:
    1. Explicit --input argument
    2. Same folder as reward.csv (if provided)
    3. Current directory
    4. policy_plots/ folder
    """
    search_paths = ["policy.txt"]
    
    if reward_path and os.path.exists(reward_path):
        map_folder = os.path.dirname(reward_path)
        if map_folder:
            search_paths.insert(0, os.path.join(map_folder, "policy.txt"))
    
    search_paths.extend([
        "policy_plots/policy.txt",
        "viz/policy.txt"
    ])
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_reward_csv(path):
    data = []
    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            row = [float(v) for v in line.split(",")]
            data.append(row)
    return np.array(data, dtype=float)


def load_start_goal(reward_path):
    if not reward_path:
        return None
    folder = os.path.dirname(reward_path)
    if not folder:
        return None
    path = os.path.join(folder, "start_goal.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        line = fp.readline().strip()
        if not line:
            return None
        parts = [int(v) for v in line.split(",")]
        if len(parts) < 4:
            return None
        return parts[0], parts[1], parts[2], parts[3]


def plot_grid(grid, title, save_path=None, show=False):
    plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap="viridis", origin="lower")
    plt.colorbar(label="policy")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_argmax(actions, title, vectors, orient, save_path=None, show=False):
    cmap = plt.get_cmap("tab10").copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    plt.figure(figsize=(6, 5))
    dir_idx = actions_to_dir(actions, vectors, orient)
    plt.imshow(dir_idx, cmap=cmap, origin="lower", vmin=0, vmax=7, interpolation="nearest")
    add_direction_legend(plt.gca(), cmap)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def action_vectors():
    positions = [
        [(0, 1), (0, 2), (0, 0), (2, 1), (2, 2), (2, 0)],
        [(0, 2), (1, 2), (0, 1), (2, 0), (2, 1), (1, 0)],
        [(1, 2), (2, 2), (0, 2), (1, 0), (2, 0), (0, 0)],
        [(2, 2), (2, 1), (1, 2), (0, 0), (1, 0), (0, 1)],
        [(2, 1), (2, 0), (2, 2), (0, 1), (0, 0), (0, 2)],
        [(2, 0), (1, 0), (2, 1), (0, 2), (0, 1), (1, 2)],
        [(1, 0), (0, 0), (2, 0), (1, 2), (0, 2), (2, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 2), (1, 2), (2, 1)],
    ]

    vectors = np.zeros((len(positions), len(positions[0]), 2), dtype=int)
    for orient, slots in enumerate(positions):
        for action, (ky, kx) in enumerate(slots):
            vectors[orient, action, 0] = kx - 1
            vectors[orient, action, 1] = ky - 1
    return vectors


def direction_order():
    return [
        ("UP", (0, 1)),
        ("UP_RIGHT", (1, 1)),
        ("RIGHT", (1, 0)),
        ("DOWN_RIGHT", (1, -1)),
        ("DOWN", (0, -1)),
        ("DOWN_LEFT", (-1, -1)),
        ("LEFT", (-1, 0)),
        ("UP_LEFT", (-1, 1)),
    ]


def direction_name(dx, dy):
    for name, vec in direction_order():
        if vec == (dx, dy):
            return name
    if (dx, dy) == (0, 0):
        return "C"
    return "?"


def direction_index(dx, dy):
    for idx, (_, vec) in enumerate(direction_order()):
        if vec == (dx, dy):
            return idx
    return -1


def actions_to_dir(actions, vectors, orient):
    lookup = np.full((vectors.shape[1],), -1, dtype=int)
    for action in range(vectors.shape[1]):
        dx = int(vectors[orient, action, 0])
        dy = int(vectors[orient, action, 1])
        lookup[action] = direction_index(dx, dy)

    dir_idx = np.full_like(actions, -1, dtype=int)
    valid = actions >= 0
    if np.any(valid):
        dir_idx[valid] = lookup[actions[valid]]
    return dir_idx


def add_direction_legend(ax, cmap):
    handles = []
    labels = []
    for idx, (name, _) in enumerate(direction_order()):
        handles.append(Patch(facecolor=cmap(idx), edgecolor="none"))
        labels.append(name)
    ax.legend(handles, labels, title="direction", loc="upper right", framealpha=0.9)


def orientation_label(orient):
    vectors = action_vectors()
    dx = int(vectors[orient, 0, 0])
    dy = int(vectors[orient, 0, 1])
    return direction_name(dx, dy)


def print_action_table():
    vectors = action_vectors()
    print("Direction legend: N(0,1) NE(1,1) E(1,0) SE(1,-1) S(0,-1) SW(-1,-1) W(-1,0) NW(-1,1)")
    print("Action table (dx, dy, dir) per orientation:")
    for orient in range(vectors.shape[0]):
        row = []
        for action in range(vectors.shape[1]):
            dx = int(vectors[orient, action, 0])
            dy = int(vectors[orient, action, 1])
            row.append(f"a{action}=({dx},{dy})[{direction_name(dx, dy)}]")
        print(f"orient {orient}: " + " ".join(row))


def gmap_table():
    return np.array(
        [
            [1, 2, 8, 1, 8, 2],
            [2, 3, 1, 2, 1, 3],
            [3, 4, 2, 3, 2, 4],
            [4, 5, 3, 4, 3, 5],
            [5, 6, 4, 5, 4, 6],
            [6, 7, 5, 6, 5, 7],
            [7, 8, 6, 7, 6, 8],
            [8, 1, 7, 8, 7, 1],
        ],
        dtype=int,
    )


def overlay_arrows(ax, actions, vectors, stride=5):
    if vectors is None:
        return
    h, w = actions.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    sample = actions[0:h:stride, 0:w:stride]
    dx = np.zeros_like(sample, dtype=float)
    dy = np.zeros_like(sample, dtype=float)

    valid = sample >= 0
    if not np.any(valid):
        return

    dx[valid] = vectors[sample[valid], 0]
    dy[valid] = vectors[sample[valid], 1]
    ax.quiver(xx, yy, dx, dy, color="white", scale=20, width=0.002, headwidth=3)


def plot_argmax_with_arrows(actions, title, vectors, orient, stride, save_path=None, show=False):
    cmap = plt.get_cmap("tab10").copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    dir_idx = actions_to_dir(actions, vectors, orient)
    ax.imshow(dir_idx, cmap=cmap, origin="lower", vmin=0, vmax=7, interpolation="nearest")
    add_direction_legend(ax, cmap)
    overlay_arrows(ax, actions, vectors[orient], stride=stride)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_argmax_mosaic(actions_list, title, vectors, stride, show_arrows, save_path=None, show=False):
    count = len(actions_list)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    cmap = plt.get_cmap("tab10").copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    im = None
    for idx, actions in enumerate(actions_list):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        dir_idx = actions_to_dir(actions, vectors, idx)
        im = ax.imshow(dir_idx, cmap=cmap, origin="lower", vmin=0, vmax=7, interpolation="nearest")
        if show_arrows:
            overlay_arrows(ax, actions, vectors[idx], stride=stride)
        ax.set_title(f"orient={orientation_label(idx)}")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(count, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    if im is not None:
        handles = [Patch(facecolor=cmap(idx), edgecolor="none") for idx in range(8)]
        labels = [name for name, _ in direction_order()]
        fig.legend(handles, labels, title="direction", loc="lower right", ncol=2, framealpha=0.9)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def choose_random_start(reward, obstacle_value, rng):
    ys, xs = np.where(reward > obstacle_value)
    if len(xs) == 0:
        return None
    idx = rng.integers(0, len(xs))
    return int(ys[idx]), int(xs[idx])


def simulate_policy(
    policy,
    reward,
    start,
    goal,
    start_orient,
    steps=500,
    obstacle_value=-1.0,
    action_mode="argmax",
    temperature=1.0,
    seed=None,
    goal_radius=0,
):
    h, w = reward.shape
    vectors = action_vectors()
    gmap = gmap_table()
    path = []
    y, x = start
    orient = start_orient
    reason = "max-steps"
    visited = set()
    
    rng = np.random.default_rng(seed)

    for _ in range(steps):
        if not (0 <= x < w and 0 <= y < h):
            reason = "start-out-of-bounds"
            break
        
        # Check goal first (highest priority)
        if max(abs(y - goal[0]), abs(x - goal[1])) <= goal_radius:
            reason = "reached-goal"
            path.append((y, x))
            break
        
        # Check for cycles (second priority)
        state = (y, x, orient)
        if state in visited:
            reason = "cycle"
            break
        visited.add(state)
        
        # Add position to path
        path.append((y, x))
        
        # Check for obstacles
        if reward[y, x] <= obstacle_value:
            reason = "hit-obstacle"
            break

        key = (orient, 0)
        if key not in policy:
            reason = "missing-policy"
            break

        actions = []
        action_count = 0
        for action in range(0, 256):
            if (orient, action) not in policy:
                break
            actions.append(policy[(orient, action)][y, x])
            action_count += 1
        if action_count == 0:
            reason = "no-actions"
            break

        values = np.array(actions, dtype=float)
        if np.all(np.isnan(values)):
            reason = "all-nan"
            break

        if action_mode == "softmax":
            action = select_action_softmax(values, temperature=temperature, rng=rng)
        else:
            action = select_action_argmax(values)
        
        dx, dy = vectors[orient, action]
        ny = y + dy
        nx = x + dx
        orient = gmap[orient, action] - 1
        if not (0 <= nx < w and 0 <= ny < h):
            reason = "step-out-of-bounds"
            break
        if reward[ny, nx] <= obstacle_value:
            reason = "hit-obstacle"
            break
        y, x = ny, nx

    return path, reason


def plot_reward_with_path(reward, path, goal, title, save_path=None, show=False):
    plt.figure(figsize=(6, 5))
    plt.imshow(reward, cmap="gray", origin="lower")
    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys, color="red", linewidth=1.5)
        plt.scatter(xs, ys, color="red", s=6, alpha=0.6)
        plt.scatter([xs[0]], [ys[0]], color="lime", s=30, label="start")
        plt.scatter([xs[-1]], [ys[-1]], color="yellow", s=30, label="end")
        plt.scatter([goal[1]], [goal[0]], color="cyan", s=40, marker="x", label="goal")
        plt.legend(loc="upper right")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def safe_nanargmax(stack):
    mask_all_nan = np.all(np.isnan(stack), axis=0)
    filled = np.where(mask_all_nan, -np.inf, stack)
    actions = np.argmax(filled, axis=0)
    actions = np.where(mask_all_nan, -1, actions)
    return actions


def select_action_argmax(values):
    """Select action using argmax (greedy)."""
    valid = ~np.isnan(values)
    if not np.any(valid):
        return -1
    return int(np.nanargmax(values))


def select_action_softmax(values, temperature=1.0, rng=None):
    """Select action using softmax sampling with temperature.
    Higher temperature = more exploration, lower = more greedy."""
    if rng is None:
        rng = np.random.default_rng()
    
    valid = ~np.isnan(values)
    if not np.any(valid):
        return -1
    
    # Clip values to prevent overflow
    v = values.copy()
    v[~valid] = -np.inf
    v_max = np.nanmax(v)
    
    # Compute softmax with temperature
    exp_v = np.exp((v - v_max) / temperature)
    exp_v[~valid] = 0
    probs = exp_v / np.sum(exp_v)
    
    # Sample action based on probabilities
    return int(rng.choice(len(values), p=probs))


def plot_all_actions_for_orient(policy, orient, title, save_path=None, show=False):
    """Plot all action Q-values for a given orientation as a mosaic."""
    grids = []
    action = 0
    while True:
        key = (orient, action)
        if key not in policy:
            break
        grids.append(policy[key])
        action += 1
    
    if not grids:
        print(f"No actions found for orient={orient}")
        return
    
    vectors = action_vectors()
    count = len(grids)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    
    vmin = min(np.nanmin(g) for g in grids)
    vmax = max(np.nanmax(g) for g in grids)
    
    for idx, grid in enumerate(grids):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        im = ax.imshow(grid, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        dx = int(vectors[orient, idx, 0])
        dy = int(vectors[orient, idx, 1])
        dir_name = direction_name(dx, dy)
        ax.set_title(f"{dir_name}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    for idx in range(count, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
    
    fig.suptitle(title)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Q-value")
    fig.subplots_adjust(right=0.9)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize policy.txt output")
    parser.add_argument("--input", default=None, help="Path to policy.txt (auto-detected if not provided)")
    parser.add_argument("--orient", type=int, default=0, help="Orientation index")
    parser.add_argument("--action", type=int, default=0, help="Action index")
    parser.add_argument("--out-dir", default="policy_plots", help="Output directory for images")
    parser.add_argument("--all", action="store_true", help="Save all orient/action grids")
    parser.add_argument("--argmax", action="store_true", help="Plot argmax action per cell for an orientation")
    parser.add_argument("--all-actions", action="store_true", help="Plot all Q-value grids for an orientation")
    parser.add_argument("--mosaic", action="store_true", help="Plot argmax mosaic for all orientations")
    parser.add_argument("--arrows", action="store_true", help="Overlay direction arrows on argmax plots")
    parser.add_argument("--stride", type=int, default=1, help="Arrow stride for argmax plots")
    parser.add_argument("--reward", default="sim_maps/reward.csv", help="Path to reward CSV for simulation")
    parser.add_argument("--simulate", action="store_true", help="Simulate policy on reward map")
    parser.add_argument("--start-x", type=int, default=None, help="Start x for simulation (overrides start_goal.csv if provided)")
    parser.add_argument("--start-y", type=int, default=None, help="Start y for simulation (overrides start_goal.csv if provided)")
    parser.add_argument("--start-orient", type=int, default=0, help="Start orientation for simulation")
    parser.add_argument("--goal-radius", type=int, default=3, help="Goal radius for simulation stop condition")
    parser.add_argument("--steps", type=int, default=500, help="Max steps for simulation")
    parser.add_argument("--print-table", action="store_true", help="Print action table")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    parser.add_argument("--action-mode", type=str, default="argmax", choices=["argmax", "softmax"], help="Action selection mode: argmax (greedy) or softmax (explore)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax sampling (higher = more exploration)")
    parser.add_argument("--seed-sim", type=int, default=None, help="Random seed for softmax sampling")
    args = parser.parse_args()

    # Auto-detect policy path if not provided
    if args.input is None:
        detected_path = find_policy_path(args.reward)
        if detected_path:
            args.input = detected_path
            print(f"Auto-detected policy from: {args.input}")
        else:
            args.input = "policy.txt"
            if not os.path.exists(args.input):
                print(f"❌ Policy file not found. Searched in:")
                print(f"   - {os.path.dirname(args.reward)}/policy.txt")
                print(f"   - policy.txt")
                print(f"   - policy_plots/policy.txt")
                return 1

    if args.print_table:
        print_action_table()
        return 0

    policy = load_policy(args.input)
    if not policy:
        print("No policy data found. Check the input format.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        for (orient, action), grid in sorted(policy.items()):
            title = f"policy orient={orient} action={action}"
            out_path = os.path.join(args.out_dir, f"policy_o{orient}_a{action}.png")
            plot_grid(grid, title, save_path=out_path, show=False)
        print(f"Saved {len(policy)} plots to {args.out_dir}")
        return 0

    vectors = action_vectors()

    if args.mosaic:
        actions_list = []
        orient = 0
        while True:
            grids = []
            for action in range(0, 256):
                key = (orient, action)
                if key not in policy:
                    break
                grids.append(policy[key])
            if not grids:
                break
            stack = np.stack(grids, axis=0)
            actions = safe_nanargmax(stack)
            actions_list.append(actions)
            orient += 1

        if not actions_list:
            print("No orientations found for mosaic")
            return 1

        out_path = os.path.join(args.out_dir, "policy_argmax_mosaic.png")
        plot_argmax_mosaic(actions_list, "policy argmax mosaic", vectors, args.stride, args.arrows, save_path=out_path, show=args.show)
        print(f"Saved mosaic plot to {out_path}")
        return 0

    if args.all_actions:
        title = f"policy all Q-values orient={orientation_label(args.orient)}"
        out_path = os.path.join(args.out_dir, f"policy_all_actions_o{args.orient}.png")
        plot_all_actions_for_orient(policy, args.orient, title, save_path=out_path, show=args.show)
        print(f"Saved all-actions plot to {out_path}")
        return 0

    if args.argmax:
        grids = []
        for action in range(0, 256):
            key = (args.orient, action)
            if key not in policy:
                break
            grids.append(policy[key])
        if not grids:
            print(f"No grids found for orient={args.orient}")
            return 1

        stack = np.stack(grids, axis=0)
        actions = safe_nanargmax(stack)
        title = f"policy argmax orient={orientation_label(args.orient)}"
        out_path = os.path.join(args.out_dir, f"policy_argmax_o{args.orient}.png")
        if args.arrows:
            plot_argmax_with_arrows(actions, title, vectors, args.orient, args.stride, save_path=out_path, show=args.show)
        else:
            plot_argmax(actions, title, vectors, args.orient, save_path=out_path, show=args.show)
        print(f"Saved argmax plot to {out_path}")
        return 0

    if args.simulate:
        reward = load_reward_csv(args.reward)
        h, w = reward.shape
        
        # Load start_goal.csv if it exists
        start_goal = load_start_goal(args.reward)
        
        # Determine start position: explicit args > start_goal.csv > defaults
        if args.start_x is not None and args.start_y is not None:
            # User explicitly provided coordinates
            start = (args.start_y, args.start_x)
            print(f"[*] Using explicit start coordinates: ({args.start_x}, {args.start_y})")
        elif start_goal:
            # Load from start_goal.csv
            sx, sy, gx, gy = start_goal
            start = (sy, sx)
            print(f"[*] Loaded start from start_goal.csv: ({sx}, {sy})")
        else:
            # Use defaults
            start = (5, 5)
            print(f"[*] Using default start coordinates: (5, 5)")
        
        # Determine goal position: start_goal.csv > defaults
        if start_goal:
            sx, sy, gx, gy = start_goal
            goal = (gy, gx)
            print(f"[*] Loaded goal from start_goal.csv: ({gx}, {gy})")
        else:
            goal = (max(0, h - 2), max(0, w - 2))
            print(f"[*] Using default goal coordinates: {goal}")
        path, reason = simulate_policy(
            policy,
            reward,
            start,
            goal,
            args.start_orient,
            steps=args.steps,
            action_mode=args.action_mode,
            temperature=args.temperature,
            seed=args.seed_sim,
            goal_radius=args.goal_radius,
        )
        out_path = os.path.join(args.out_dir, "policy_path.png")
        plot_reward_with_path(reward, path, goal, "policy rollout", save_path=out_path, show=args.show)
        print(f"Saved rollout plot to {out_path}")
        if not path:
            print(f"Simulation produced no path: {reason}")
        else:
            print(f"Simulation stop reason: {reason}")
        return 0

    key = (args.orient, args.action)
    if key not in policy:
        print(f"Key not found: orient={args.orient} action={args.action}")
        return 1

    title = f"policy orient={args.orient} action={args.action}"
    out_path = os.path.join(args.out_dir, f"policy_o{args.orient}_a{args.action}.png")
    plot_grid(policy[key], title, save_path=out_path, show=args.show)
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
