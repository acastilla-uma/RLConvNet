#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib.pyplot as plt


def read_csv(path):
    grid = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grid.append([float(v) for v in line.split(",")])
    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Visualize obstacle and reward maps from generated map folders"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="sim_maps/map_d0.35_s42_sm3_cr2",
        help="Path to map folder (created by map_generator.py)",
    )
    parser.add_argument(
        "--show-reward",
        action="store_true",
        help="Display reward map in addition to obstacles",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save combined visualization to this path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures instead of only saving",
    )
    args = parser.parse_args()

    # Resolve map folder path
    if not os.path.isdir(args.folder):
        print(f"❌ Map folder not found: {args.folder}")
        print("Available maps in sim_maps/:")
        if os.path.isdir("sim_maps"):
            for subfolder in os.listdir("sim_maps"):
                subfolder_path = os.path.join("sim_maps", subfolder)
                if os.path.isdir(subfolder_path):
                    print(f"  - {subfolder}")
        sys.exit(1)

    obstacles_path = os.path.join(args.folder, "obstacles.csv")
    reward_path = os.path.join(args.folder, "reward.csv")
    start_goal_path = os.path.join(args.folder, "start_goal.csv")

    # Check that required files exist
    if not os.path.exists(obstacles_path):
        print(f"❌ obstacles.csv not found in {args.folder}")
        sys.exit(1)

    obstacles = read_csv(obstacles_path)
    reward = read_csv(reward_path) if os.path.exists(reward_path) else None

    sx = sy = gx = gy = None
    if os.path.exists(start_goal_path):
        sg = read_csv(start_goal_path)
        if sg and len(sg[0]) >= 4:
            sx, sy, gx, gy = map(int, sg[0][:4])

    # Determine layout
    if args.show_reward and reward:
        rows, cols = 1, 2
        fig_size = (12, 5)
    else:
        rows, cols = 1, 1
        fig_size = (6, 6)

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if not isinstance(axes, list):
        axes = [axes]

    # Plot obstacles
    ax = axes[0]
    ax.set_title("Obstacles (black) and free space (white)")
    ax.imshow(obstacles, cmap="gray_r", interpolation="nearest", origin="upper")

    if sx is not None:
        ax.plot(sx, sy, "go", markersize=8, label="start")
    if gx is not None:
        ax.plot(gx, gy, "r*", markersize=12, label="goal")

    if sx is not None or gx is not None:
        ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot reward map if requested
    if args.show_reward and reward and len(axes) > 1:
        ax = axes[1]
        ax.set_title("Reward Map")
        im = ax.imshow(reward, cmap="viridis", interpolation="nearest", origin="upper")
        plt.colorbar(im, ax=ax, label="Reward")

        if sx is not None:
            ax.plot(sx, sy, "go", markersize=8, label="start")
        if gx is not None:
            ax.plot(gx, gy, "r*", markersize=12, label="goal")

        if sx is not None or gx is not None:
            ax.legend(loc="upper right")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()

    # Save if requested
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"✓ Saved visualization to: {args.save}")

    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"✓ Visualized map from: {args.folder}")
    if sx is not None and gx is not None:
        print(f"  Start: ({sx}, {sy}), Goal: ({gx}, {gy})")


if __name__ == "__main__":
    main()
