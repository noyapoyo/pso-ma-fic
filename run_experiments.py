"""
=============================================================================
run_experiments.py - Experiment 1: PSO vs Memetic-PSO vs FG-PSO vs FGDS
=============================================================================

四種方法的公平比較（FFE budget = 800），每個 (method, image) 跑 5 runs。

■ 方法:
  - pso             : Standard PSO (Muruganandham 2010 baseline)
  - memetic_pso     : PSO + Local Search (isometry + spatial)
  - feature_guided_pso : FG-PSO, K=40 feature-guided candidate pool + PSO + LS
  - feature_exhaustive : FGDS (pairwise), K=40 feature pool + exhaustive (no PSO)

■ 重點結論（從 narrative 來）:
  - PSO → Memetic PSO: LS 有幫助但更慢
  - Memetic PSO → FG-PSO: feature 縮搜索空間讓 PSO 更有效
  - FG-PSO → FGDS: 在 K=40 的候選池中，直接 exhaustive 比 PSO 更好

■ 用法:
    python run_experiments.py \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 800 --n-runs 5 --output-dir results_exp1

    # 快速測試單張圖
    python run_experiments.py --image 0007 --n-runs 1 \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 800 --output-dir results_exp1_test

■ 畫圖:
    python plot_convergence.py results_exp1/

設定檔結構：
    configs/global.yml           全域設定 (seed 等)
    configs/pso.yml              PSO 參數
    configs/memetic_pso.yml      Memetic PSO 參數
    configs/feature_guided_pso.yml  FG-PSO 參數 (top_k=40)
    configs/feature_exhaustive.yml  FGDS 參數 (top_k=40)
=============================================================================
"""

import argparse
import os
import sys
import yaml
import csv
import numpy as np
from collections import defaultdict
from datetime import datetime

import fic_core as core
from encoders.pso import encode_pso
from encoders.memetic_pso import encode_memetic_pso
from encoders.feature_guided_pso import encode_feature_guided_pso
from encoders.feature_exhaustive import encode_feature_exhaustive


# Experiment 1 methods (K=40 for feature-based methods, set in configs/)
METHODS = {
    'pso':                encode_pso,
    'memetic_pso':        encode_memetic_pso,
    'feature_guided_pso': encode_feature_guided_pso,
    'feature_exhaustive': encode_feature_exhaustive,
}

CONFIGS_DIR = 'configs'


# =============================================================================
# Config loader
# =============================================================================

def load_yaml(path):
    """讀取 YAML 檔，回傳 dict（檔案不存在則回傳空 dict）。"""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_global_config(path=None):
    """讀取全域設定。"""
    if path is None:
        path = os.path.join(CONFIGS_DIR, 'global.yml')
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using built-in defaults.")
        return {}
    return load_yaml(path)


def load_method_config(method_name, configs_dir=CONFIGS_DIR):
    """讀取特定方法的設定。"""
    path = os.path.join(configs_dir, f"{method_name}.yml")
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using encoder defaults.")
        return {}
    return load_yaml(path)


def build_encoder_kwargs(method_name, global_cfg, method_cfg):
    """
    將 global + method config 合併成傳給 encoder 的 kwargs。

    處理：
      - ls_strategies: list → tuple (encoder 介面要求)
      - 注入 ffe_budget_per_block (除 full_search 外)
      - 注入 seed
      - 過濾掉 GA-only 或不相容的參數
    """
    kwargs = dict(method_cfg)

    # ls_strategies: YAML 是 list，encoder 收 tuple
    if 'ls_strategies' in kwargs:
        kwargs['ls_strategies'] = tuple(kwargs['ls_strategies'] or [])

    # FFE budget (Full Search 不受限)
    if method_name != 'full_search':
        budget = global_cfg.get('ffe_budget_per_block', None)
        if budget is not None:
            kwargs['ffe_budget_per_block'] = budget

            # ■ 公平比較核心：當 FFE budget 設定時，budget 必須是唯一的終止條件
            #   - max_iter 設為極大值（safety net），不會先於 budget 觸發
            #   - early_stop_patience 設為極大值，不會提前終止
            #   這確保所有演算法在相同的 FFE budget 下被公平比較。
            kwargs['max_iter'] = 9999
            kwargs['early_stop_patience'] = 9999

    # --no-early-stop: 獨立於 budget，可單獨使用
    if global_cfg.get('no_early_stop', False):
        kwargs['early_stop_patience'] = 9999

    # Seed (從 global config 注入，若 method config 沒寫)
    if 'seed' not in kwargs and 'seed' in global_cfg:
        kwargs['seed'] = global_cfg['seed']

    # 過濾 GA-only 參數（避免傳給不支援的 encoder 出錯）
    ga_only_keys = {'generation', 'run', 'tournament_size', 'crossover_rate',
                    'mutation_rate', 'elitism', 'domain_mutation_radius'}
    if method_name != 'ga':
        for k in list(kwargs.keys()):
            if k in ga_only_keys:
                kwargs.pop(k)

    # Full Search 是確定性的，不需要 seed 和 FFE budget
    if method_name == 'full_search':
        kwargs.pop('seed', None)
        kwargs.pop('ffe_budget_per_block', None)

    return kwargs


# =============================================================================
# 影像收集
# =============================================================================

def collect_images(image_dir, single_image=None):
    """收集要跑的影像清單。"""
    if single_image:
        for ext in ['', '.png', '.jpg', '.bmp', '.tif', '.tiff', '.jpeg']:
            cand = os.path.join(image_dir, single_image + ext)
            if os.path.exists(cand):
                return [cand]
        if os.path.exists(single_image):
            return [single_image]
        print(f"Error: image '{single_image}' not found.")
        sys.exit(1)

    if not os.path.exists(image_dir):
        print(f"Error: {image_dir}/ not found.")
        sys.exit(1)

    files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    if not files:
        print(f"No images found in {image_dir}/")
        sys.exit(1)
    return files


# =============================================================================
# 輸出表格
# =============================================================================

def print_aggregated_table(aggregated, n_runs):
    """印出方法 × 影像的彙總表，多次 run 取 mean ± std。"""
    print("\n" + "=" * 110)
    if n_runs > 1:
        print(f"  AGGREGATED RESULTS (mean ± std over {n_runs} runs)")
    else:
        print(f"  RESULTS")
    print("=" * 110)
    print(f"  {'Image':<16} {'Method':<16} {'Time(s)':>14} "
          f"{'PSNR(dB)':>16} {'CR':>8} {'AvgMSE':>16} {'Evals':>14}")
    print("-" * 110)

    for (method, image), runs in sorted(aggregated.items()):
        if n_runs == 1:
            s = runs[0]
            evals = s.get('n_evaluations', 0)
            evals_str = f"{evals:,}" if evals else "-"
            print(f"  {image:<16} {method:<16} "
                  f"{s['encoding_time_sec']:>14.2f} "
                  f"{s['psnr_actual']:>16.2f} "
                  f"{s['compression_ratio']:>7.1f}:1 "
                  f"{s['mse_mean']:>16.4f} "
                  f"{evals_str:>14}")
        else:
            t = np.array([r['encoding_time_sec'] for r in runs])
            p = np.array([r['psnr_actual']       for r in runs])
            m = np.array([r['mse_mean']          for r in runs])
            e = int(np.mean([r.get('n_evaluations', 0) for r in runs]))

            print(f"  {image:<16} {method:<16} "
                  f"{t.mean():>7.2f}±{t.std():<5.2f} "
                  f"{p.mean():>8.2f}±{p.std():<6.2f} "
                  f"{runs[0]['compression_ratio']:>7.1f}:1 "
                  f"{m.mean():>8.4f}±{m.std():<6.4f} "
                  f"{e:>14,}")
    print("=" * 110)


def save_results_csv(all_results, output_dir):
    """把所有 run 的結果存成 CSV，方便後續分析。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"experiment_results_{timestamp}.csv")

    # 收集所有可能的欄位
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    # 標準欄位優先
    priority = ['image', 'method', 'run_idx', 'encoding_time_sec',
                'psnr_actual', 'psnr_db', 'mse_mean', 'mse_max',
                'compression_ratio', 'n_evaluations',
                'n_evals_global', 'n_evals_ls',
                'ls_triggers', 'ls_improvements', 'fic_file_size_kb']
    keys = [k for k in priority if k in all_keys]
    keys += sorted([k for k in all_keys if k not in priority])

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in keys}
            # list/dict 轉字串
            for k, v in row.items():
                if isinstance(v, (list, tuple, dict)):
                    row[k] = str(v)
            writer.writerow(row)

    print(f"\n  Saved CSV: {path}")
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FIC experiment runner with YAML configs and FFE budget",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--methods', nargs='+', default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help='要執行的方法')
    parser.add_argument('--image', type=str, default=None,
                        help='單張影像名稱 (預設跑 images/ 下全部)')
    # Image arguments
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--image-size', type=int, default=None,
                        help='影像 resize 邊長 (覆寫 global.yml)')
    # FIC arguments
    parser.add_argument('--range-size', type=int, default=None,
                        help='Range block size (覆寫 global.yml)')
    parser.add_argument('--domain-size', type=int, default=None,
                        help='Domain block size (覆寫 global.yml)')
    parser.add_argument('--domain-stride', type=int, default=None,
                        help='Domain block stride (覆寫 global.yml)')
    parser.add_argument('--decode-iter', type=int, default=None,
                        help='Decode iteration 次數 (覆寫 global.yml)')
    # Output & runs
    parser.add_argument('--output-dir', type=str, default='results_exp1',
                        help='Output directory (default: results_exp1)')
    parser.add_argument('--global-config', type=str, default=None,
                        help='全域設定檔路徑 (預設 configs/global.yml)')
    parser.add_argument('--configs-dir', type=str, default=CONFIGS_DIR)
    parser.add_argument('--n-runs', type=int, default=None,
                        help='每個方法×影像跑幾次 (覆寫 global.yml)')
    # FFE budget & early stopping
    parser.add_argument('--ffe-budget', type=int, default=None,
                        help='FFE budget per range block (覆寫 global.yml)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='禁用 early stopping (設 patience=9999)')
    args = parser.parse_args()

    # 讀取全域設定
    global_cfg = load_global_config(args.global_config)

    # CLI 參數覆寫 global config (CLI > YAML > 程式預設)
    if args.image_size    is not None: global_cfg['image_size']         = args.image_size
    if args.range_size    is not None: global_cfg['range_size']         = args.range_size
    if args.domain_size   is not None: global_cfg['domain_size']        = args.domain_size
    if args.domain_stride is not None: global_cfg['domain_stride']      = args.domain_stride
    if args.decode_iter   is not None: global_cfg['decode_iterations']  = args.decode_iter
    if args.ffe_budget    is not None: global_cfg['ffe_budget_per_block'] = args.ffe_budget
    if args.no_early_stop:             global_cfg['no_early_stop']      = True

    output_dir = args.output_dir
    n_runs = args.n_runs if args.n_runs is not None else global_cfg.get('n_runs', 1)
    image_size = global_cfg.get('image_size', 256)

    # 收集影像
    images = collect_images(args.image_dir, args.image)

    # 印出實驗配置
    print(f"\n{'=' * 72}")
    print(f"  FIC Experiment Configuration")
    print(f"{'=' * 72}")
    print(f"  Methods:           {args.methods}")
    print(f"  Images:            {[os.path.basename(p) for p in images]}")
    print(f"  Image size:        {image_size}x{image_size}")
    print(f"  Output dir:        {output_dir}/")
    print(f"  Runs per (m, i):   {n_runs}")
    print(f"  Global config:     {args.global_config or 'configs/global.yml'}")
    print(f"  FFE budget/block:  {global_cfg.get('ffe_budget_per_block', 'unlimited')}")
    print(f"  Early stopping:    {'disabled' if global_cfg.get('no_early_stop') else 'enabled (auto-disabled when budget is set)'}")
    print(f"  Block params:      "
          f"range={global_cfg.get('range_size', 8)}, "
          f"domain={global_cfg.get('domain_size', 16)}, "
          f"stride={global_cfg.get('domain_stride', 8)}")
    print(f"{'=' * 72}\n")

    # === 執行實驗 ===
    all_results = []
    aggregated = defaultdict(list)  # (method, image) → list of stats

    for method_name in args.methods:
        method_cfg = load_method_config(method_name, args.configs_dir)
        encoder_kwargs = build_encoder_kwargs(method_name, global_cfg, method_cfg)
        encoder_fn = METHODS[method_name]

        for image_path in images:
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            for run_idx in range(n_runs):
                # 每次 run 用不同 seed = base_seed + run_idx
                base_seed = encoder_kwargs.get('seed', 42)
                run_kwargs = dict(encoder_kwargs)
                if 'seed' in encoder_kwargs:
                    run_kwargs['seed'] = base_seed + run_idx

                if n_runs > 1:
                    print(f"\n>>> Run {run_idx+1}/{n_runs}  "
                          f"method={method_name}  image={image_name}  "
                          f"seed={run_kwargs['seed']}")

                stats = core.run_pipeline(
                    encoder_fn, image_path,
                    method_name=method_name,
                    output_dir=output_dir,
                    image_size=image_size,
                    range_size=global_cfg.get('range_size', 8),
                    domain_size=global_cfg.get('domain_size', 16),
                    domain_stride=global_cfg.get('domain_stride', 8),
                    decode_iterations=global_cfg.get('decode_iterations', 20),
                    save_fic=global_cfg.get('save_fic', True),
                    save_outputs=(run_idx == 0),
                    **run_kwargs,
                )
                stats['run_idx'] = run_idx
                all_results.append(stats)
                aggregated[(method_name, image_name)].append(stats)

    # === 輸出表格 + CSV ===
    print_aggregated_table(aggregated, n_runs)
    save_results_csv(all_results, output_dir)

    # === 儲存 convergence data ===
    from convergence import save_convergence
    # 按 (image, run_idx) 分組，收集各方法的 convergence curve
    conv_groups = defaultdict(dict)  # (image, run_idx) → {method: curve}
    for r in all_results:
        curve = r.get('convergence_curve')
        if curve:
            key = (r['image'], r.get('run_idx', 0))
            conv_groups[key][r['method']] = curve
    for (image_name, run_idx), curves in conv_groups.items():
        path = save_convergence(curves, output_dir, image_name, run_idx)
        print(f"  Saved convergence: {path}")


if __name__ == "__main__":
    main()
