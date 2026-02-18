#!/usr/bin/env python3
"""Shared utilities for batch experiments and pipeline."""

import os
import subprocess
import re


def find_solver_executable(solver_exe=None):
    """Find solver executable in multiple locations.
    
    Search order:
    1. Explicit --solver-exe parameter
    2. Current directory
    3. core/bin/ directory
    4. PATH
    """
    if solver_exe and os.path.exists(solver_exe):
        return solver_exe
    
    search_paths = [
        "rl_convnet_simple.exe",
        os.path.join("core", "bin", "rl_convnet_simple.exe"),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    # Try PATH
    result = subprocess.run(
        ["where" if os.name == "nt" else "which", "rl_convnet_simple.exe"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip().split("\n")[0]
    
    return None


def parse_solver_output(output):
    """Parse solver output to extract iteration count and convergence status."""
    iters = None
    converged = False
    convergence_message = "Desconocido"
    
    # Look for: "RLConvNet simple demo. V(goal)=... iters=500 (converged) tol=0.001"
    match = re.search(r'iters=(\d+)\s+\((converged|maxed)\)', output)
    if match:
        iters = int(match.group(1))
        converged = (match.group(2) == 'converged')
    
    # Extract convergence message
    if 'Converged at iteration' in output:
        conv_match = re.search(r'Converged at iteration (\d+) with tol=([0-9.e-]+)', output)
        if conv_match:
            convergence_message = f"Convergió en iteración {conv_match.group(1)} (tol={conv_match.group(2)})"
    elif 'Warning: value iteration hit k-max' in output:
        warn_match = re.search(r'hit k-max=(\d+) without reaching tol=([0-9.e-]+)', output)
        if warn_match:
            convergence_message = f"Alcanzó k-max={warn_match.group(1)} sin llegar a tol={warn_match.group(2)}"
    
    return iters, converged, convergence_message


def run_train(map_folder, tol, k_max, solver_exe):
    """Run training and return results.
    
    Args:
        map_folder: Path to map folder (can be name or full path)
        tol: Convergence tolerance
        k_max: Maximum iterations
        solver_exe: Path to solver executable
    
    Returns:
        dict with training results
    """
    # Normalize map_folder
    if not os.path.isabs(map_folder) and not map_folder.startswith("sim_maps"):
        map_folder = os.path.join("sim_maps", map_folder)
    
    reward_csv = os.path.join(map_folder, "reward.csv")
    
    if not os.path.exists(reward_csv):
        print(f"  ERROR: reward.csv not found: {reward_csv}")
        return {
            'tol': tol,
            'k_max': k_max,
            'iters': None,
            'converged': False,
            'success': False
        }
    
    if not os.path.exists(solver_exe):
        print(f"  ERROR: Solver not found: {solver_exe}")
        return {
            'tol': tol,
            'k_max': k_max,
            'iters': None,
            'converged': False,
            'success': False
        }
    
    cmd = [
        solver_exe,
        "--reward", reward_csv,
        "--tol", str(tol),
        "--k-max", str(k_max)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore')
        output = result.stdout + result.stderr
        
        # Print relevant lines only
        for line in output.split('\n'):
            if any(kw in line for kw in ['Policy saved', 'Warning:', 'RLConvNet', 'Converged']):
                print(f"  {line.strip()}")
        
        iters, converged, convergence_msg = parse_solver_output(output)
        success = iters is not None
        
        return {
            'tol': tol,
            'k_max': k_max,
            'iters': iters,
            'converged': converged,
            'convergence_message': convergence_msg,
            'success': success
        }
    except subprocess.TimeoutExpired:
        print(f"  Warning: Training timed out")
        return {
            'tol': tol,
            'k_max': k_max,
            'iters': None,
            'converged': False,
            'convergence_message': 'Timeout de entrenamiento',
            'success': False
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'tol': tol,
            'k_max': k_max,
            'iters': None,
            'converged': False,
            'convergence_message': f'Error: {str(e)}',
            'success': False
        }
