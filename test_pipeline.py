#!/usr/bin/env python3
"""
Test pipeline for training and simulating RL policies on different maps.
Automates the process: generate map -> train solver -> simulate policy
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


def train_policy(map_folder, solver_exe="rl_convnet_simple.exe", tol=1e-4, k_max=50000):
    """Train policy for a map using the solver."""
    reward_csv = os.path.join(map_folder, "reward.csv")
    
    if not os.path.exists(reward_csv):
        print(f"  ❌ reward.csv not found in {map_folder}")
        return None
    
    # Check if solver exists
    if not os.path.exists(solver_exe):
        print(f"  ❌ Solver executable not found: {solver_exe}")
        return None
    
    cmd = [
        solver_exe,
        "--reward", reward_csv,
        "--tol", str(tol),
        "--k-max", str(k_max)
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ❌ Solver failed with return code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
            return None
        
        # Extract convergence info from output
        output = result.stdout + result.stderr
        print(f"  {output.strip()}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  ❌ Solver timed out (>300s)")
        return None
    except Exception as e:
        print(f"  ❌ Error running solver: {e}")
        return None


def simulate_policy(map_folder, steps=10000, action_mode="argmax", temperature=1.0):
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
        "--temperature", str(temperature)
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  ⚠ Simulation returned non-zero code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
        
        output = result.stdout + result.stderr
        print(f"  {output.strip()}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Simulation timed out (>60s)")
        return None
    except Exception as e:
        print(f"  ⚠ Error running simulation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test pipeline: train and simulate policies on maps"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["list", "train", "simulate", "all"],
        default="list",
        help="Action to perform"
    )
    parser.add_argument(
        "--map",
        type=str,
        help="Specific map to use (from sim_maps/ folder), e.g., map_d0.35_s42_sm3_cr2"
    )
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
        "--all-maps",
        action="store_true",
        help="Run action on all available maps"
    )
    args = parser.parse_args()

    # Determine which maps to process
    if args.all_maps:
        maps = list_available_maps()
        if not maps:
            print("❌ No maps found in sim_maps/")
            sys.exit(1)
    elif args.map:
        map_path = os.path.join("sim_maps", args.map) if not args.map.startswith("sim_maps") else args.map
        if not os.path.isdir(map_path):
            print(f"❌ Map folder not found: {map_path}")
            sys.exit(1)
        maps = [args.map if args.map.startswith("sim_maps") else f"sim_maps/{args.map}"]
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
            print("❌ Please specify --map or --all-maps")
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
            print("🔧 Training policy...")
            train_policy(map_path, tol=args.tol, k_max=args.k_max)

        if args.action in ("simulate", "all"):
            print("🎮 Simulating policy...")
            simulate_policy(
                map_path,
                steps=args.steps,
                action_mode=args.action_mode,
                temperature=args.temperature
            )


if __name__ == "__main__":
    main()
