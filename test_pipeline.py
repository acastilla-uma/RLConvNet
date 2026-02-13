#!/usr/bin/env python3
"""
Test pipeline for training and simulating RL policies.
Automates: generate map -> train solver -> simulate policy
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def list_available_maps(maps_dir="sim_maps"):
    """List all available map folders."""
    if not os.path.isdir(maps_dir):
        return []
    
    maps = [d for d in os.listdir(maps_dir) 
            if os.path.isdir(os.path.join(maps_dir, d)) and d.startswith("map_")]
    return sorted(maps)


def generate_map(
    density,
    seed,
    smooth=3,
    carve=2,
    width=100,
    height=100,
    start="5,5",
    goal="94,94",
):
    """Generate a map using map_generator.py."""
    cmd = [
        "python", "sim/map_generator.py",
        "--density", str(density),
        "--seed", str(seed),
        "--smooth", str(smooth),
        "--carve", str(carve),
        "--width", str(width),
        "--height", str(height),
        "--start", str(start),
        "--goal", str(goal),
    ]
    
    print(f"  Running: {' '.join(cmd[:5])}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  ❌ Map generation failed")
            if result.stderr:
                print(f"  {result.stderr[:200]}")
            return None
        
        output = result.stdout
        # Extract map folder name from output
        for line in output.split('\n'):
            if 'Generated map in:' in line:
                # Extract folder path
                parts = line.split()
                map_path = parts[-1]
                print(f"  ✓ Generated: {os.path.basename(map_path)}")
                return map_path
        
        return None
    except subprocess.TimeoutExpired:
        print(f"  ❌ Map generation timed out")
        return None
    except Exception as e:
        print(f"  ❌ Error generating map: {e}")
        return None


def train_policy(map_folder, solver_exe="rl_convnet_simple.exe", tol=1e-4, k_max=50000):
    """Train policy for a map using the solver."""
    reward_csv = os.path.join(map_folder, "reward.csv")
    
    if not os.path.exists(reward_csv):
        print(f"  ❌ reward.csv not found in {map_folder}")
        return None
    
    if not os.path.exists(solver_exe):
        print(f"  ❌ Solver executable not found: {solver_exe}")
        return None
    
    cmd = [
        solver_exe,
        "--reward", reward_csv,
        "--tol", str(tol),
        "--k-max", str(k_max)
    ]
    
    print(f"  Running solver...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        output = result.stdout + result.stderr
        # Extract important lines
        for line in output.split('\n'):
            if 'Converged at' in line or 'Warning:' in line or 'saved to' in line:
                print(f"  {line.strip()}")
        
        return True
    except subprocess.TimeoutExpired:
        print(f"  ❌ Solver timed out (>300s)")
        return None
    except Exception as e:
        print(f"  ❌ Error running solver: {e}")
        return None


def simulate_policy(map_folder, steps=10000, action_mode="argmax", temperature=1.0, goal_radius=0):
    """Simulate policy for a map."""
    reward_csv = os.path.join(map_folder, "reward.csv")
    
    if not os.path.exists(reward_csv):
        print(f"  ❌ reward.csv not found in {map_folder}")
        return None
    
    cmd = [
        "python", "viz/plot_policy.py",
        "--simulate",
        "--steps", str(steps),
        "--reward", reward_csv,
        "--action-mode", action_mode,
        "--temperature", str(temperature),
        "--goal-radius", str(goal_radius),
        "--out-dir", map_folder,
    ]
    
    print(f"  Running simulation...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if 'Auto-detected' in line or 'Saved' in line or 'reason:' in line:
                print(f"  {line.strip()}")
        
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Simulation timed out (>60s)")
        return None
    except Exception as e:
        print(f"  ⚠ Error running simulation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: generate map -> train solver -> simulate"
    )
    
    # Action
    parser.add_argument(
        "--action",
        type=str,
        choices=["list", "train", "simulate", "all"],
        default="list",
        help="Action to perform"
    )
    
    # Existing map or generate new
    parser.add_argument(
        "--map",
        type=str,
        help="Existing map (from sim_maps/), e.g., map_d0.35_s42_sm3_cr2"
    )
    
    # Map generation parameters
    parser.add_argument(
        "--density",
        type=float,
        help="Generate new map with density (0-1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for map generation"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Smoothing passes for obstacle generation"
    )
    parser.add_argument(
        "--carve",
        type=int,
        default=2,
        help="Carve radius for path generation"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="5,5",
        help="Start position as 'x,y'"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="94,94",
        help="Goal position as 'x,y'"
    )
    
    # Solver parameters
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for solver"
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=50000,
        help="Maximum iterations for solver"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Maximum simulation steps"
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default="argmax",
        choices=["argmax", "softmax"],
        help="Action selection mode during simulation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax action selection"
    )
    parser.add_argument(
        "--goal-radius",
        type=int,
        default=0,
        help="Goal radius for simulation stop condition"
    )
    
    # Batch processing
    parser.add_argument(
        "--all-maps",
        action="store_true",
        help="Run action on all available maps"
    )
    
    args = parser.parse_args()

    # If generating new map
    if args.density is not None or args.seed is not None:
        if args.density is None or args.seed is None:
            print("ERROR: Both --density and --seed required for map generation")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(
            f"Generating map: density={args.density}, seed={args.seed}, "
            f"start={args.start}, goal={args.goal}"
        )
        print(f"{'='*60}")
        
        map_path = generate_map(
            density=args.density,
            seed=args.seed,
            smooth=args.smooth,
            carve=args.carve,
            start=args.start,
            goal=args.goal,
        )
        
        if not map_path:
            print("ERROR: Map generation failed")
            sys.exit(1)
        
        # Execute full pipeline on generated map
        print(f"\n{'='*60}")
        print(f"Training on generated map")
        print(f"{'='*60}")
        
        print("[*] Training policy...")
        train_policy(map_path, tol=args.tol, k_max=args.k_max)
        
        print("\n[*] Simulating policy...")
        simulate_policy(
            map_path,
            steps=args.steps,
            action_mode=args.action_mode,
            temperature=args.temperature,
            goal_radius=args.goal_radius,
        )
        
        print(f"\n[+] Pipeline complete!")
        print(f"   Map: {map_path}")
        print(f"   Policy: {os.path.join(map_path, 'policy.txt')}")
        print(f"   Plot: policy_plots/policy_path.png")
        return

    # Determine which existing maps to process
    if args.all_maps:
        maps = list_available_maps()
        if not maps:
            print("ERROR: No maps found in sim_maps/")
            sys.exit(1)
    elif args.map:
        map_path = os.path.join("sim_maps", args.map) if not args.map.startswith("sim_maps") else args.map
        if not os.path.isdir(map_path):
            print(f"ERROR: Map folder not found: {map_path}")
            sys.exit(1)
        maps = [map_path]
    else:
        # List action
        if args.action == "list":
            maps = list_available_maps()
            if not maps:
                print("No maps found in sim_maps/")
                sys.exit(0)
            print("Available maps:")
            for m in maps:
                map_path = os.path.join("sim_maps", m)
                files = os.listdir(map_path)
                print(f"  {m}")
                print(f"    Files: {', '.join(files)}")
            sys.exit(0)
        else:
            print("ERROR: Please specify --map, --all-maps, or use --density --seed to generate")
            sys.exit(1)

    # Ensure maps paths are in proper format
    maps = [os.path.join("sim_maps", m) if not m.startswith("sim_maps") else m for m in maps]

    # Execute action on each map
    for map_path in maps:
        map_name = os.path.basename(map_path)
        print(f"\n{'='*60}")
        print(f"Processing map: {map_name}")
        print(f"{'='*60}")

        if args.action in ("train", "all"):
            print("[*] Training policy...")
            train_policy(map_path, tol=args.tol, k_max=args.k_max)

        if args.action in ("simulate", "all"):
            print("[*] Simulating policy...")
            simulate_policy(
                map_path,
                steps=args.steps,
                action_mode=args.action_mode,
                temperature=args.temperature,
                goal_radius=args.goal_radius
            )


if __name__ == "__main__":
    main()
