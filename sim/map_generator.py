#!/usr/bin/env python3
import argparse
import math
import os
import random
from collections import deque

import matplotlib.pyplot as plt


def smooth_grid(grid, w, h, passes):
    for _ in range(passes):
        new_grid = [row[:] for row in grid]
        for y in range(h):
            for x in range(w):
                acc = 0.0
                cnt = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= nx < w and 0 <= ny < h:
                            acc += grid[ny][nx]
                            cnt += 1
                new_grid[y][x] = acc / cnt
        grid = new_grid
    return grid


def generate_obstacles(w, h, density, smooth_passes, seed):
    rnd = random.Random(seed)
    grid = [[rnd.random() for _ in range(w)] for _ in range(h)]
    grid = smooth_grid(grid, w, h, smooth_passes)

    obstacles = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            obstacles[y][x] = 1 if grid[y][x] < density else 0
    return obstacles


def carve_path(obstacles, w, h, start, goal, carve_radius, rng=None):
    # Carve a random path with guaranteed connectivity to goal.
    if rng is None:
        rng = random.Random()
    
    x, y = start
    gx, gy = goal
    visited = set()
    visited.add((x, y))
    
    def clear_cell(cx, cy):
        for dy in range(-carve_radius, carve_radius + 1):
            for dx in range(-carve_radius, carve_radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    obstacles[ny][nx] = 0
    
    # Phase 1: Random exploration
    max_steps = w * h
    step_count = 0
    clear_cell(x, y)
    
    while step_count < max_steps:
        direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        x += direction[0]
        y += direction[1]
        
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        clear_cell(x, y)
        visited.add((x, y))
        step_count += 1
        
        if (x, y) == (gx, gy):
            return  # Already at goal
        
        if len(visited) > (w * h * 0.3):
            break
    
    # Phase 2: Biased random walk to goal (guarantees arrival)
    # 70% chance to move towards goal, 30% random direction
    max_reach_steps = w + h + 500
    reach_step = 0
    
    while (x, y) != (gx, gy) and reach_step < max_reach_steps:
        dx_goal = 1 if gx > x else (-1 if gx < x else 0)
        dy_goal = 1 if gy > y else (-1 if gy < y else 0)
        
        # Biased choice towards goal
        if rng.random() < 0.7:
            # Move towards goal, but randomize which axis
            if rng.random() < 0.5 and dx_goal != 0:
                x += dx_goal
            elif dy_goal != 0:
                y += dy_goal
            elif dx_goal != 0:
                x += dx_goal
        else:
            # Random detour
            direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            x += direction[0]
            y += direction[1]
        
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        clear_cell(x, y)
        visited.add((x, y))
        reach_step += 1
    
    clear_cell(gx, gy)


def distance_to_obstacles(obstacles, w, h):
    dist = [[-1 for _ in range(w)] for _ in range(h)]
    q = deque()
    for y in range(h):
        for x in range(w):
            if obstacles[y][x] == 1:
                dist[y][x] = 0
                q.append((x, y))

    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and dist[ny][nx] < 0:
                dist[ny][nx] = dist[y][x] + 1
                q.append((nx, ny))
    return dist


def build_reward_map(
    obstacles,
    w,
    h,
    obstacle_penalty,
    free_reward,
    reward_range,
    goal,
    goal_reward,
    goal_radius,
):
    reward_min, reward_max = reward_range
    reward = [[free_reward for _ in range(w)] for _ in range(h)]

    dist = distance_to_obstacles(obstacles, w, h)
    max_dist = max(1, max(max(row) for row in dist))

    for y in range(h):
        for x in range(w):
            if obstacles[y][x] == 1:
                reward[y][x] = obstacle_penalty
            else:
                t = dist[y][x] / max_dist
                reward[y][x] = reward_min + t * (reward_max - reward_min)
    if goal is not None:
        gx, gy = goal
        for dy in range(-goal_radius, goal_radius + 1):
            for dx in range(-goal_radius, goal_radius + 1):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < w and 0 <= ny < h and obstacles[ny][nx] == 0:
                    reward[ny][nx] = max(reward[ny][nx], goal_reward)
    return reward


def write_csv(path, grid):
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            f.write(",".join(str(v) for v in row) + "\n")


def visualize_obstacles(obstacles, w, h, sx, sy, gx, gy, output_path):
    """Generate and save an obstacle map visualization."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Obstacle Map")
    ax.imshow(obstacles, cmap="gray_r", interpolation="nearest", origin="upper")
    
    if sx is not None and sy is not None:
        ax.plot(sx, sy, "go", markersize=10, label="Start")
    if gx is not None and gy is not None:
        ax.plot(gx, gy, "r*", markersize=15, label="Goal")
    
    ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def generate_folder_name(args):
    """Generate descriptive folder name from map generation parameters."""
    # Format: map_d{density}_s{seed}_sm{smooth}_cr{carve}
    name = f"map_d{args.density}_s{args.seed}_sm{args.smooth}_cr{args.carve}"
    return name


def main():
    parser = argparse.ArgumentParser(description="Random map generator with simulated obstacles")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--density", type=float, default=0.35, help="Obstacle density (0..1)")
    parser.add_argument("--smooth", type=int, default=3, help="Smoothing passes for obstacle blobs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--carve", type=int, default=2, help="Radius for guaranteed free corridor")
    parser.add_argument("--out", type=str, default="sim_maps")
    parser.add_argument("--obstacle-penalty", type=float, default=-1.0)
    parser.add_argument("--free-reward", type=float, default=0.0)
    parser.add_argument("--start", type=str, default="5,5")
    parser.add_argument("--goal", type=str, default="50,50")
    parser.add_argument("--reward-min", type=float, default=0)
    parser.add_argument("--reward-max", type=float, default=1)
    parser.add_argument("--goal-reward", type=float, default=5.0, help="Reward at/near goal")
    args = parser.parse_args()

    w, h = args.width, args.height
    sx, sy = (int(v) for v in args.start.split(","))
    gx, gy = (int(v) for v in args.goal.split(","))

    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        raise SystemExit("Start/goal out of bounds")

    # Generate map
    obstacles = generate_obstacles(w, h, args.density, args.smooth, args.seed)
    rng = random.Random(args.seed + 1)
    carve_path(obstacles, w, h, (sx, sy), (gx, gy), args.carve, rng)
    reward = build_reward_map(
        obstacles,
        w,
        h,
        args.obstacle_penalty,
        args.free_reward,
        (args.reward_min, args.reward_max),
        (gx, gy),
        args.goal_reward,
        1,  # goal_radius fixed to 1
    )

    # Create descriptive folder with map parameters
    folder_name = generate_folder_name(args)
    map_folder = os.path.join(args.out, folder_name)
    os.makedirs(map_folder, exist_ok=True)

    # Save CSV files
    write_csv(os.path.join(map_folder, "obstacles.csv"), obstacles)
    write_csv(os.path.join(map_folder, "reward.csv"), reward)
    write_csv(os.path.join(map_folder, "start_goal.csv"), [[sx, sy, gx, gy]])

    # Generate and save obstacle visualization
    obstacles_img_path = os.path.join(map_folder, "obstacles.png")
    visualize_obstacles(obstacles, w, h, sx, sy, gx, gy, obstacles_img_path)

    print(f"Generated map in: {map_folder}")
    print(f"  - obstacles.csv (0 free, 1 obstacle)")
    print(f"  - reward.csv (float reward per cell)")
    print(f"  - start_goal.csv (sx,sy,gx,gy)")
    print(f"  - obstacles.png (visualization)")


if __name__ == "__main__":
    main()
