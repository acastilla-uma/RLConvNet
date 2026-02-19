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
import json
from datetime import datetime
from html import escape
from pathlib import Path
from utils import find_solver_executable, run_train


def run_train_batch(map_name, tol, k_max, solver_exe):
    """Run training for batch experiments."""
    print(f"\n{'='*70}")
    print(f"Training: tol={tol:.1e}, k-max={k_max}")
    print(f"{'='*70}")
    
    map_folder = os.path.join("sim_maps", map_name)
    return run_train(map_folder, tol, k_max, solver_exe)


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


def generate_html_report(map_name, experiments, report_path, notes=None, embed_images=True):
    """Generate HTML report with all results and visualizations."""
    
    # Get map info
    map_parts = map_name.split('_')
    density = map_parts[1][1:] if len(map_parts) > 1 else "?"
    seed = map_parts[2][1:] if len(map_parts) > 2 else "?"
    
    safe_notes = escape(notes) if notes else ""
    notes_json = json.dumps({"notes": notes if notes else ""})
    notes_html = (
        "<p><strong>Notas:</strong></p>"
        "<div class=\"notes-block\">"
        f"<textarea id=\"notes\" class=\"notes-area\" rows=\"4\">{safe_notes}</textarea>"
        "<div class=\"notes-actions\">"
        "<button id=\"save-notes\" class=\"notes-btn\" type=\"button\">Guardar notas</button>"
        "<span class=\"notes-help\">Se guardan en el archivo HTML al hacer click en Guardar.</span>"
        "</div>"
        "</div>"
        f"<div id=\"notes-data\" style=\"display:none;\">{notes_json}</div>"
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
    
    def img_src_for(data_value):
        if embed_images and data_value:
            return f"data:image/png;base64,{data_value}"
        return None
    
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
        plot_src = img_src_for(exp.get('plot_data'))
        if plot_src:
            html += f"""
                <div>
                    <h5 style="text-align: center; color: #34495e; margin-bottom: 10px;">Trayectoria Simulada</h5>
                    <img src="{plot_src}" alt="Policy Path Exp {exp['exp_num']}" style="width: 100%;">
                </div>
"""
        
        # Add mosaic plot
        mosaic_src = img_src_for(exp.get('mosaic_data'))
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
            const notesArea = document.getElementById("notes");
            const saveBtn = document.getElementById("save-notes");
            const notesData = document.getElementById("notes-data");

            if (!notesArea || !notesData) return;

            // Load initial notes from embedded JSON
            try {{
                const data = JSON.parse(notesData.textContent);
                if (data.notes) {{
                    notesArea.value = data.notes;
                }}
            }} catch (e) {{
                console.error("Error parsing notes data:", e);
            }}

            if (saveBtn) {{
                saveBtn.addEventListener("click", function() {{
                    // Update the embedded notes data
                    const updatedData = {{ notes: notesArea.value }};
                    notesData.textContent = JSON.stringify(updatedData);
                    
                    // Get the full HTML with updated notes
                    const htmlContent = document.documentElement.outerHTML;
                    
                    // Create blob and download
                    const blob = new Blob([htmlContent], {{ type: 'text/html;charset=utf-8' }});
                    const link = document.createElement('a');
                    const url = URL.createObjectURL(blob);
                    link.setAttribute('href', url);
                    link.setAttribute('download', '{map_name}_experiment_report.html');
                    link.style.visibility = 'hidden';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    alert('Notas guardadas. El archivo HTML se ha descargado.');
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
        default=None,
        help="Map name (e.g., map_d0.7_s42_sm3_cr2). Required if --all-maps not specified."
    )
    
    parser.add_argument(
        "--all-maps",
        action="store_true",
        help="Run experiments for all maps in sim_maps/ directory"
    )
    
    parser.add_argument(
        "--tol",
        type=float,
        nargs='+',
        default=[1e-3],
        help="Tolerance values to test (e.g., 1e-3 1e-4 1e-5)"
    )
    
    parser.add_argument(
        "--k-max",
        type=int,
        nargs='+',
        default=[50, 100, 1000, 5000],
        help="Maximum iteration values to test (e.g., 1000 5000 10000)"
    )
    
    parser.add_argument(
        "--goal-radius",
        type=int,
        default=1,
        help="Goal radius for simulation (default: 1)"
    )

    parser.add_argument(
        "--solver-exe",
        type=str,
        default=None,
        help="Path to solver executable (auto-detected if not provided)"
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
    
    # Validate map arguments
    if not args.map and not args.all_maps:
        print(f"ERROR: Specify either --map <name> or --all-maps")
        sys.exit(1)
    
    # Find solver executable
    solver_exe = find_solver_executable(args.solver_exe)
    if not solver_exe:
        print(f"ERROR: Solver executable not found. Specify with --solver-exe")
        sys.exit(1)
    print(f"[*] Using solver: {solver_exe}")
    
    # Get list of maps to process
    if args.all_maps:
        sim_maps_dir = "sim_maps"
        if not os.path.isdir(sim_maps_dir):
            print(f"ERROR: sim_maps directory not found")
            sys.exit(1)
        maps_to_process = [d for d in os.listdir(sim_maps_dir) 
                           if os.path.isdir(os.path.join(sim_maps_dir, d))]
        maps_to_process.sort()
        if not maps_to_process:
            print(f"ERROR: No maps found in sim_maps/ directory")
            sys.exit(1)
    else:
        map_path = os.path.join("sim_maps", args.map)
        if not os.path.isdir(map_path):
            print(f"ERROR: Map folder not found: {map_path}")
            sys.exit(1)
        maps_to_process = [args.map]
    
    # Process each map
    for map_name in maps_to_process:
        print(f"\n{'='*70}")
        print(f"BATCH EXPERIMENTS")
        print(f"{'='*70}")
        print(f"Map: {map_name}")
        print(f"Tolerances: {', '.join(f'{t:.1e}' for t in args.tol)}")
        print(f"K-max values: {', '.join(str(k) for k in args.k_max)}")
        print(f"Total experiments: {len(args.tol) * len(args.k_max)}")
        
        # Run all experiments
        experiments = []
        exp_num = 0
        map_path = os.path.join("sim_maps", map_name)
        
        for tol in args.tol:
            for k_max in args.k_max:
                exp_num += 1
                
                # Train policy
                train_result = run_train_batch(map_name, tol, k_max, solver_exe)
                
                # Simulate policy immediately after training
                sim_result = run_simulate(map_name, args.goal_radius, f" (exp {exp_num})")
                
                # Load trajectory image into memory and remove the file
                map_folder = os.path.join("sim_maps", map_name)
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
                mosaic_data = None
                if args.generate_mosaic:
                    mosaic_path = os.path.join(map_folder, "policy_argmax_mosaic.png")
                    reward_path = os.path.join(map_folder, "reward.csv")
                    cmd = [
                        "python", "viz/plot_policy.py",
                        "--input", os.path.join(map_folder, "policy.txt"),
                        "--reward", reward_path,
                        "--mosaic",
                        "--arrows",
                        "--stride", "5",
                        "--out-dir", map_folder
                    ]
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
                        if os.path.exists(mosaic_path):
                            try:
                                with open(mosaic_path, "rb") as img_file:
                                    mosaic_data = base64.b64encode(img_file.read()).decode("ascii")
                            except OSError as exc:
                                print(f"  Warning: could not read mosaic image: {exc}")
                            try:
                                os.remove(mosaic_path)
                            except OSError:
                                pass
                    except Exception as e:
                        print(f"  Warning: mosaic generation failed: {e}")
                
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
                    'plot_data': plot_data,
                    'mosaic_data': mosaic_data
                }
                experiments.append(experiment)
        
        # Generate HTML report
        if args.report:
            report_path = args.report
        else:
            report_path = os.path.join(map_path, f"{map_name}.html")
        
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
            map_name,
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
            mosaic_count = sum(1 for e in experiments if e.get('mosaic_data'))
            print(f"\n[+] Mosaicos de políticas generados: {mosaic_count}/{len(experiments)}")
        
        print(f"\n[+] Informe HTML: {report_path}")
        print(f"\n[+] Experimentos completados para {map_name}")
    
    print(f"\n{'='*70}")
    print(f"[✓] Todos los mapas procesados")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
