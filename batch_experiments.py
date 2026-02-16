#!/usr/bin/env python3
"""
Batch experiments: Train policies with different parameters and generate report.

Usage:
    python batch_experiments.py --map map_d0.7_s42_sm3_cr2 \
        --tol 1e-3 1e-4 1e-5 1e-6 \
        --k-max 500 1000 5000 10000
"""

import argparse
import os
import subprocess
import sys
import re
import shutil
import base64
from datetime import datetime
from html import escape
from pathlib import Path


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


def run_train(map_name, tol, k_max, solver_exe="rl_convnet_simple.exe"):
    """Run training and return results."""
    print(f"\n{'='*70}")
    print(f"Training: tol={tol:.1e}, k-max={k_max}")
    print(f"{'='*70}")
    
    reward_csv = os.path.join("sim_maps", map_name, "reward.csv")
    
    if not os.path.exists(reward_csv):
        print(f"  ERROR: reward.csv not found: {reward_csv}")
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
        
        success = iters is not None  # Consider success if we got iterations
        
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


def run_simulate(map_name, goal_radius, output_suffix=""):
    """Run simulation and return path to generated plot."""
    print(f"\n{'='*70}")
    print(f"Simulating trajectory{output_suffix}")
    print(f"{'='*70}")
    
    cmd = [
        "python", "test_pipeline.py",
        "--action", "simulate",
        "--map", map_name,
        "--goal-radius", str(goal_radius)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
        output = result.stdout + result.stderr
        
        # Print relevant lines
        for line in output.split('\n'):
            if any(kw in line for kw in ['Auto-detected', 'Saved', 'reason:', 'Simulating']):
                print(f"  {line.strip()}")
        
        # Extract stop reason
        stop_reason = "unknown"
        match = re.search(r'reason:\s+(\S+)', output)
        if match:
            stop_reason = match.group(1)
        
        # Path to generated plot
        map_path = os.path.join("sim_maps", map_name)
        plot_path = os.path.join(map_path, "policy_path.png")
        
        # Translate stop reasons to Spanish
        reason_translation = {
            'reached-goal': 'Llegó a la meta',
            'hit-obstacle': 'Chocó con obstáculo',
            'cycle': 'Ciclo detectado',
            'max-steps': 'Máximo de pasos alcanzado',
            'unknown': 'Desconocido'
        }
        
        return {
            'success': result.returncode == 0 and os.path.exists(plot_path),
            'plot_path': plot_path if os.path.exists(plot_path) else None,
            'stop_reason': stop_reason,
            'stop_reason_spanish': reason_translation.get(stop_reason, stop_reason)
        }
    except subprocess.TimeoutExpired:
        print(f"  Warning: Simulation timed out")
        return {'success': False, 'plot_path': None, 'stop_reason': 'timeout', 'stop_reason_spanish': 'Timeout'}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'success': False, 'plot_path': None, 'stop_reason': 'error', 'stop_reason_spanish': 'Error'}


def generate_policy_mosaic(map_name):
    """Generate policy mosaic visualization."""
    print(f"\n{'='*70}")
    print(f"Generating policy mosaic")
    print(f"{'='*70}")
    
    map_path = os.path.join("sim_maps", map_name)
    policy_path = os.path.join(map_path, "policy.txt")
    
    if not os.path.exists(policy_path):
        print(f"  Warning: Policy not found: {policy_path}")
        return None
    
    mosaic_output = os.path.join(map_path, "policy_argmax_mosaic.png")
    
    cmd = [
        "python", "viz/plot_policy.py",
        "--input", policy_path,
        "--mosaic",
        "--arrows",
        "--stride", "5",
        "--out-dir", map_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
        output = result.stdout + result.stderr
        
        # Print output
        for line in output.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
        
        if result.returncode == 0 and os.path.exists(mosaic_output):
            print(f"  [+] Mosaic saved to: {mosaic_output}")
            return mosaic_output
        else:
            print(f"  Warning: Mosaic generation failed (returncode={result.returncode})")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def generate_policy_mosaic_custom(map_name, output_filename):
    """Generate policy mosaic visualization with custom filename."""
    map_path = os.path.join("sim_maps", map_name)
    policy_path = os.path.join(map_path, "policy.txt")
    
    if not os.path.exists(policy_path):
        print(f"  Warning: Policy not found: {policy_path}")
        return None
    
    # Generate mosaic with default name first
    cmd = [
        "python", "viz/plot_policy.py",
        "--input", policy_path,
        "--mosaic",
        "--arrows",
        "--stride", "5",
        "--out-dir", map_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
        
        default_mosaic = os.path.join(map_path, "policy_argmax_mosaic.png")
        custom_mosaic = os.path.join(map_path, output_filename)
        
        if os.path.exists(default_mosaic):
            shutil.copy2(default_mosaic, custom_mosaic)
            print(f"  [+] Saved mosaic: {output_filename}")
            return custom_mosaic
        else:
            print(f"  Warning: Mosaic generation failed")
            return None
    except Exception as e:
        print(f"  ERROR generating mosaic: {e}")
        return None


def generate_html_report(map_name, experiments, report_path, notes=None, embed_images=True):
    """Generate HTML report with all results and visualizations."""
    
    # Get map info
    map_parts = map_name.split('_')
    density = map_parts[1][1:] if len(map_parts) > 1 else "?"
    seed = map_parts[2][1:] if len(map_parts) > 2 else "?"
    
    safe_notes = escape(notes) if notes else ""
    notes_html = (
        "<p><strong>Notas:</strong></p>"
        "<div class=\"notes-block\">"
        f"<textarea id=\"notes\" class=\"notes-area\" rows=\"4\">{safe_notes}</textarea>"
        "<div class=\"notes-actions\">"
        "<button id=\"save-notes\" class=\"notes-btn\" type=\"button\">Guardar notas</button>"
        "<span class=\"notes-help\">Se guardan en este navegador (localStorage).</span>"
        "</div>"
        "</div>"
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Experimentos - {map_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .notes {{
            margin-top: 8px;
            padding: 10px 12px;
            background-color: #ffffff;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
            white-space: pre-wrap;
        }}
        .notes-block {{
            margin-top: 8px;
        }}
        .notes-area {{
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            border: 1px solid #dcdcdc;
            border-radius: 4px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
        }}
        .notes-actions {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 8px;
        }}
        .notes-btn {{
            padding: 6px 12px;
            background-color: #3498db;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
        }}
        .notes-btn:hover {{
            background-color: #2e86c1;
        }}
        .notes-help {{
            color: #7f8c8d;
            font-size: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .converged {{
            color: #27ae60;
            font-weight: bold;
        }}
        .maxed {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .success {{
            color: #27ae60;
        }}
        .failure {{
            color: #e74c3c;
        }}
        .visualization {{
            margin: 30px 0;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .visualization h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h4 {{
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <h1>📊 Informe de Experimentos de Entrenamiento</h1>
    
    <div class="metadata">
        <p><strong>Mapa:</strong> {map_name}</p>
        <p><strong>Densidad de obstáculos:</strong> {density}</p>
        <p><strong>Semilla:</strong> {seed}</p>
        <p><strong>Fecha de generación:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Total de experimentos:</strong> {len(experiments)}</p>
        {notes_html}
    </div>
    
    <h2>📈 Resumen de Resultados</h2>
    
    <div class="stats">
"""
    
    # Calculate statistics
    successful = sum(1 for exp in experiments if exp['success'])
    converged_count = sum(1 for exp in experiments if exp.get('converged', False))
    
    html += f"""
        <div class="stat-card">
            <h4>Experimentos Exitosos</h4>
            <div class="value">{successful}/{len(experiments)}</div>
        </div>
        <div class="stat-card">
            <h4>Convergieron</h4>
            <div class="value">{converged_count}/{len(experiments)}</div>
        </div>
"""
    
    if converged_count > 0:
        avg_iters = sum(exp['iters'] for exp in experiments if exp.get('converged', False)) / converged_count
        html += f"""
        <div class="stat-card">
            <h4>Iteraciones Promedio (Convergidos)</h4>
            <div class="value">{avg_iters:.0f}</div>
        </div>
"""
    
    html += """
    </div>
    
    <h2>Tabla de Experimentos</h2>
    
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Tolerancia (tol)</th>
                <th>Máx Iteraciones (k-max)</th>
                <th>Iteraciones Realizadas</th>
                <th>Estado Convergencia</th>
                <th>Razón Simulación</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for idx, exp in enumerate(experiments, 1):
        status_class = "converged" if exp.get('converged') else "maxed"
        iters_text = str(exp['iters']) if exp['iters'] is not None else "N/A"
        reason_text = exp.get('stop_reason_spanish', 'Desconocido')
        convergence_text = exp.get('convergence_message', 'Desconocido')
        
        html += f"""
            <tr>
                <td>{idx}</td>
                <td>{exp['tol']:.1e}</td>
                <td>{exp['k_max']}</td>
                <td>{iters_text}</td>
                <td class="{status_class}">{convergence_text}</td>
                <td>{reason_text}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
"""
    
    # Add visualizations section
    html += """
    <h2>Visualizaciones</h2>
"""

    def img_src_for(path_value, data_value):
        if embed_images and data_value:
            return f"data:image/png;base64,{data_value}"
        if not path_value or not os.path.exists(path_value):
            return None
        if embed_images:
            try:
                with open(path_value, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            except OSError:
                return None
        rel_path = os.path.relpath(path_value, os.path.dirname(report_path))
        return rel_path.replace(os.sep, "/")
    
    # Add trajectory plots and mosaics for each experiment
    html += """
    <div class="visualization">
        <h3>Resultados por Experimento</h3>
"""
    
    for exp in experiments:
        exp_title = f"Exp #{exp['exp_num']}: tol={exp['tol']:.1e}, k-max={exp['k_max']}, iters={exp['iters'] or 'N/A'}"
        reason = exp.get('stop_reason_spanish', 'Desconocido')
        
        html += f"""
        <div style="margin: 30px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
            <h4 style="color: #2c3e50; margin-top: 0;">{exp_title}</h4>
            <p style="color: #7f8c8d;">Razón de detención: {reason}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
"""
        
        # Add trajectory plot
        plot_src = img_src_for(exp.get('plot_path'), exp.get('plot_data'))
        if plot_src:
            html += f"""
                <div>
                    <h5 style="text-align: center; color: #34495e; margin-bottom: 10px;">Trayectoria Simulada</h5>
                    <img src="{plot_src}" alt="Policy Path Exp {exp['exp_num']}" style="width: 100%;">
                </div>
"""
        
        # Add mosaic plot
        mosaic_src = img_src_for(exp.get('mosaic_path'), exp.get('mosaic_data'))
        if mosaic_src:
            html += f"""
                <div>
                    <h5 style="text-align: center; color: #34495e; margin-bottom: 10px;">Mosaico de Políticas</h5>
                    <img src="{mosaic_src}" alt="Policy Mosaic Exp {exp['exp_num']}" style="width: 100%;">
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
"""
    

    
    html += f"""
    <div class="footer">
        <p>Generado automáticamente por batch_experiments.py</p>
        <p>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <script>
        (function() {{
            const notesKey = "experiment_notes_{map_name}";
            const notesArea = document.getElementById("notes");
            const saveBtn = document.getElementById("save-notes");

            if (!notesArea) return;

            const cached = localStorage.getItem(notesKey);
            if (cached !== null) {{
                notesArea.value = cached;
            }}

            if (saveBtn) {{
                saveBtn.addEventListener("click", function() {{
                    localStorage.setItem(notesKey, notesArea.value);
                    notesArea.textContent = notesArea.value;
                }});
            }}
        }})();
    </script>
</body>
</html>
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n[+] Informe HTML generado: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiments: Train with different parameters and generate report"
    )
    
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Map name (e.g., map_d0.7_s42_sm3_cr2)"
    )
    
    parser.add_argument(
        "--tol",
        type=float,
        nargs='+',
        default=[1e-3, 1e-4, 1e-5],
        help="Tolerance values to test (e.g., 1e-3 1e-4 1e-5)"
    )
    
    parser.add_argument(
        "--k-max",
        type=int,
        nargs='+',
        default=[1000, 5000, 10000],
        help="Maximum iteration values to test (e.g., 1000 5000 10000)"
    )
    
    parser.add_argument(
        "--goal-radius",
        type=int,
        default=1,
        help="Goal radius for simulation (default: 1)"
    )
    
    parser.add_argument(
        "--generate-mosaic",
        action="store_true",
        help="Generate policy mosaic visualization"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Output path for HTML report (default: sim_maps/<map>/experiment_report.html)"
    )

    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional notes to include in the HTML report"
    )

    
    args = parser.parse_args()
    
    # Verify map exists
    map_path = os.path.join("sim_maps", args.map)
    if not os.path.isdir(map_path):
        print(f"ERROR: Map folder not found: {map_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"BATCH EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Map: {args.map}")
    print(f"Tolerances: {', '.join(f'{t:.1e}' for t in args.tol)}")
    print(f"K-max values: {', '.join(str(k) for k in args.k_max)}")
    print(f"Total experiments: {len(args.tol) * len(args.k_max)}")
    
    # Run all experiments
    experiments = []
    exp_num = 0
    
    for tol in args.tol:
        for k_max in args.k_max:
            exp_num += 1
            
            # Train policy
            train_result = run_train(args.map, tol, k_max)
            
            # Simulate policy immediately after training
            sim_result = run_simulate(args.map, args.goal_radius, f" (exp {exp_num})")
            
            # Load trajectory image into memory and remove the file
            map_folder = os.path.join("sim_maps", args.map)
            source_path = os.path.join(map_folder, "policy_path.png")
            plot_data = None
            if os.path.exists(source_path):
                try:
                    with open(source_path, "rb") as img_file:
                        plot_data = base64.b64encode(img_file.read()).decode("ascii")
                except OSError as exc:
                    print(f"  Warning: could not read trajectory image: {exc}")
                try:
                    os.remove(source_path)
                except OSError:
                    pass
            
            # Generate mosaic for this experiment if requested
            mosaic_path = None
            mosaic_data = None
            if args.generate_mosaic:
                mosaic_filename = f"policy_mosaic_tol{tol:.0e}_kmax{k_max}.png"
                mosaic_path = generate_policy_mosaic_custom(args.map, mosaic_filename)
                if mosaic_path and os.path.exists(mosaic_path):
                    try:
                        with open(mosaic_path, "rb") as img_file:
                            mosaic_data = base64.b64encode(img_file.read()).decode("ascii")
                    except OSError as exc:
                        print(f"  Warning: could not read mosaic image: {exc}")
                    try:
                        os.remove(mosaic_path)
                    except OSError:
                        pass
                default_mosaic = os.path.join(map_folder, "policy_argmax_mosaic.png")
                if os.path.exists(default_mosaic):
                    try:
                        os.remove(default_mosaic)
                    except OSError:
                        pass
            
            # Combine results
            experiment = {
                'exp_num': exp_num,
                'tol': tol,
                'k_max': k_max,
                'iters': train_result['iters'],
                'converged': train_result['converged'],
                'convergence_message': train_result.get('convergence_message', 'Desconocido'),
                'success': train_result['success'],
                'stop_reason': sim_result['stop_reason'],
                'stop_reason_spanish': sim_result['stop_reason_spanish'],
                'plot_path': None,
                'plot_data': plot_data,
                'mosaic_path': None,
                'mosaic_data': mosaic_data
            }
            experiments.append(experiment)
    
    # Generate HTML report
    if args.report:
        report_path = args.report
    else:
        report_path = os.path.join(map_path, f"{args.map}.html")
    
    notes_value = args.notes
    if notes_value is None:
        notes_path = os.path.join(os.path.dirname(report_path), "notes.txt")
        if os.path.exists(notes_path):
            try:
                with open(notes_path, "r", encoding="utf-8", errors="ignore") as f:
                    notes_value = f.read().strip()
            except OSError as exc:
                print(f"Warning: could not read notes file: {notes_path} ({exc})")

    generate_html_report(
        args.map,
        experiments,
        report_path,
        notes=notes_value
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RESUMEN")
    print(f"{'='*70}")
    print(f"Total de experimentos: {len(experiments)}")
    print(f"Exitosos: {sum(1 for e in experiments if e['success'])}")
    print(f"Convergieron: {sum(1 for e in experiments if e.get('converged', False))}")
    print(f"Alcanzaron k-max: {sum(1 for e in experiments if not e.get('converged', False))}")
    
    # Count simulation results
    reached_goal = sum(1 for e in experiments if e.get('stop_reason') == 'reached-goal')
    if reached_goal > 0:
        print(f"\n[+] Simulaciones que llegaron a la meta: {reached_goal}/{len(experiments)}")
    
    if args.generate_mosaic:
        mosaic_count = sum(1 for e in experiments if e.get('mosaic_path'))
        print(f"\n[+] Mosaicos de políticas generados: {mosaic_count}/{len(experiments)}")
    
    print(f"\n[+] Informe HTML: {report_path}")
    print(f"\n[+] Experimentos completados")


if __name__ == "__main__":
    main()
