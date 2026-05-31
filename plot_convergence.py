#!/usr/bin/env python3
"""
=============================================================================
plot_convergence.py - Convergence Curve Visualization
=============================================================================

讀取各實驗產出的 convergence JSON 檔，畫出 PSNR vs FFE 收斂曲線。

用法：
    # 畫單張影像的 convergence curve
    python plot_convergence.py results_exp1/convergence_0007_run0.json

    # 畫某個實驗目錄下所有影像的平均 convergence curve
    python plot_convergence.py results_exp1/
    python plot_convergence.py results_exp2/
    python plot_convergence.py results_exp3/
    python plot_convergence.py results_exp4/

    # 指定輸出檔名
    python plot_convergence.py results_exp1/ -o exp1_convergence.png
=============================================================================
"""

import json
import os
import sys
import glob
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# Method display styles
# Each entry: label (legend text), color, ls (linestyle), marker
METHOD_STYLES = {
    # === Experiment 1: PSO variants ===
    'pso':                {'label': 'PSO',                  'color': '#1f77b4', 'ls': '--',  'marker': 'o'},
    'memetic_pso':        {'label': 'Memetic PSO',          'color': '#2ca02c', 'ls': '-.',  'marker': '^'},
    'feature_guided_pso': {'label': 'FG-PSO (K=40)',        'color': '#9467bd', 'ls': '-',   'marker': '*'},
    'feature_exhaustive': {'label': 'FGDS Exhaustive (K=40)', 'color': '#d62728', 'ls': '-', 'marker': 'P'},

    # === Experiment 2: Three-way FGDS comparison ===
    'fgds_pairwise': {'label': 'FGDS Pairwise (K=80, no KDTree)', 'color': '#ff7f0e', 'ls': '--', 'marker': 's'},
    'fgds_kdtree':   {'label': 'FGDS + KDTree (K=80)',             'color': '#1f77b4', 'ls': '-',  'marker': 'P'},
    'fgds_ga':       {'label': 'FGDS + KDTree + GA (K=80)',        'color': '#d62728', 'ls': '-',  'marker': '*'},

    # === Experiment 3: K sensitivity ===
    'fgds_k5':   {'label': 'FGDS K=5',   'color': '#bcbd22', 'ls': ':',  'marker': '.'},
    'fgds_k10':  {'label': 'FGDS K=10',  'color': '#17becf', 'ls': ':',  'marker': 'v'},
    'fgds_k20':  {'label': 'FGDS K=20',  'color': '#e377c2', 'ls': '--', 'marker': '<'},
    'fgds_k40':  {'label': 'FGDS K=40',  'color': '#8c564b', 'ls': '--', 'marker': 'P'},
    'fgds_k80':  {'label': 'FGDS K=80',  'color': '#d62728', 'ls': '-',  'marker': '>'},
    'fgds_k160': {'label': 'FGDS K=160', 'color': '#1f77b4', 'ls': '-',  'marker': 'h'},

    # === Experiment 4: Random vs FGDS baseline ===
    'random_k80': {'label': 'Random-80 Exhaustive',       'color': '#7f7f7f', 'ls': '--', 'marker': 'x'},
    # fgds_k80 already defined above

    # === Others / legacy ===
    'ppso':               {'label': 'PPSO',              'color': '#ff7f0e', 'ls': '--', 'marker': 's'},
    'memetic_ppso':       {'label': 'Memetic PPSO',      'color': '#8c1a1a', 'ls': '-.', 'marker': 'D'},
    'feature_kdtree':     {'label': 'FGDS KDTree',       'color': '#8c564b', 'ls': '-',  'marker': 'P'},
    'fgds_equal_weights': {'label': 'FGDS (equal w)',    'color': '#8c564b', 'ls': '--', 'marker': 'P'},
    'fgds_ga_weights':    {'label': 'FGDS (GA w)',       'color': '#e41a1c', 'ls': '-',  'marker': '*'},
}


def load_convergence_files(path):
    """
    載入 convergence JSON 檔案。

    Args:
        path: 單一 JSON 檔或目錄

    Returns:
        list of dict: [{method: [{'ffe', 'avg_mse', 'psnr_db'}, ...]}, ...]
    """
    if os.path.isfile(path):
        with open(path) as f:
            return [json.load(f)]

    # 目錄：載入所有 convergence_*.json
    files = sorted(glob.glob(os.path.join(path, 'convergence_*.json')))
    if not files:
        print(f"No convergence_*.json files found in {path}/")
        sys.exit(1)

    data = []
    for fp in files:
        with open(fp) as f:
            data.append(json.load(f))
    return data


def average_curves(all_data):
    """
    將多個 convergence data (多張影像/多個 run) 平均成一組 curve。

    Returns:
        dict: {method: {'ffe': [...], 'psnr_mean': [...], 'psnr_std': [...]}}
    """
    # 收集每個方法在各 checkpoint 的 PSNR
    method_values = {}  # method → {ffe → [psnr1, psnr2, ...]}

    for data in all_data:
        for method, curve in data.items():
            if method not in method_values:
                method_values[method] = {}
            for point in curve:
                ffe = point['ffe']
                psnr = point['psnr_db']
                if ffe not in method_values[method]:
                    method_values[method][ffe] = []
                method_values[method][ffe].append(psnr)

    # 計算 mean +/- std
    result = {}
    for method, ffe_dict in method_values.items():
        ffes = sorted(ffe_dict.keys())
        psnr_mean = [float(np.mean(ffe_dict[f])) for f in ffes]
        psnr_std = [float(np.std(ffe_dict[f])) for f in ffes]
        result[method] = {
            'ffe': ffes,
            'psnr_mean': psnr_mean,
            'psnr_std': psnr_std,
        }
    return result


def plot_single(data, title="Convergence Curve", output_path=None):
    """畫單一 convergence data (一張影像, 一個 run)。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for method, curve in data.items():
        style = METHOD_STYLES.get(method, {
            'label': method, 'color': 'gray', 'ls': '-', 'marker': '.'
        })
        ffes = [p['ffe'] for p in curve]
        psnrs = [p['psnr_db'] for p in curve]
        ax.plot(ffes, psnrs,
                label=style['label'],
                color=style['color'],
                linestyle=style['ls'],
                marker=style['marker'],
                markersize=4,
                markevery=max(1, len(ffes) // 10),
                linewidth=1.5)

    ax.set_xlabel('FFE (Fitness Function Evaluations per block)', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_path or 'convergence_curve.png'
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_averaged(averaged, title="Average Convergence Curve", output_path=None):
    """畫平均 convergence curve (含 std 帶狀區間)。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for method, vals in averaged.items():
        style = METHOD_STYLES.get(method, {
            'label': method, 'color': 'gray', 'ls': '-', 'marker': '.'
        })
        ffes = np.array(vals['ffe'])
        means = np.array(vals['psnr_mean'])
        stds = np.array(vals['psnr_std'])

        ax.plot(ffes, means,
                label=style['label'],
                color=style['color'],
                linestyle=style['ls'],
                marker=style['marker'],
                markersize=4,
                markevery=max(1, len(ffes) // 10),
                linewidth=1.5)

        # std 帶狀區間
        if np.any(stds > 0):
            ax.fill_between(ffes, means - stds, means + stds,
                            color=style['color'], alpha=0.15)

    ax.set_xlabel('FFE (Fitness Function Evaluations per block)', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_path or 'convergence_curve_avg.png'
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_per_image(all_data, output_dir, filenames=None):
    """為每張影像畫一張 convergence curve，再畫一張平均。"""
    # 逐張影像
    for i, data in enumerate(all_data):
        if filenames and i < len(filenames):
            name = os.path.splitext(os.path.basename(filenames[i]))[0]
            name = name.replace('convergence_', '').replace('_run0', '')
        else:
            name = f"image_{i}"

        out_path = os.path.join(output_dir, f"convergence_{name}.png")
        plot_single(data, title=f"Convergence - {name}", output_path=out_path)

    # 平均
    if len(all_data) > 1:
        averaged = average_curves(all_data)
        out_path = os.path.join(output_dir, "convergence_average.png")
        n = len(all_data)
        plot_averaged(averaged,
                      title=f"Average Convergence Curve ({n} images)",
                      output_path=out_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from experiment results",
    )
    parser.add_argument('path', help='JSON file or directory with convergence_*.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output image path (for single file mode)')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        # 單一檔案
        with open(args.path) as f:
            data = json.load(f)
        title = os.path.splitext(os.path.basename(args.path))[0]
        title = title.replace('convergence_', '').replace('_run0', '')
        plot_single(data, title=f"Convergence - {title}",
                    output_path=args.output)
    else:
        # 目錄
        files = sorted(glob.glob(os.path.join(args.path, 'convergence_*.json')))
        if not files:
            print(f"No convergence_*.json found in {args.path}/")
            sys.exit(1)

        all_data = []
        for fp in files:
            with open(fp) as f:
                all_data.append(json.load(f))

        plot_per_image(all_data, args.path, filenames=files)


if __name__ == "__main__":
    main()
