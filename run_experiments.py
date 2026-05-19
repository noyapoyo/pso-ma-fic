"""
=============================================================================
run_experiment.py - 統一的 FIC 實驗執行器 (YAML config + FFE budget)
=============================================================================

讀取 configs/ 下的 YAML 設定檔執行實驗。支援 FFE budget 公平比較。

用法：
    # 跑所有方法（讀 configs/*.yml）
    python run_experiment.py

    # 只跑特定方法
    python run_experiment.py --methods pso ppso memetic_ppso

    # 只跑單張影像
    python run_experiment.py --image cameraman

    # 跑多次 run（每個方法跑 N 次取平均、std）
    python run_experiment.py --n-runs 5

    # 用自訂全域設定
    python run_experiment.py --global-config configs/global_strict.yml

設定檔結構：
    configs/global.yml       全域設定 (FFE budget, 影像參數, seed, n_runs)
    configs/<method>.yml     方法特定參數 (pop_size, max_iter, ...)

擴充新方法：
    1. 在 encoders/ 新增 your_method.py
    2. 在 METHODS 字典加一行
    3. 建立 configs/your_method.yml
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
from encoders.full_search import encode_full_search
from encoders.pso import encode_pso
from encoders.ppso import encode_ppso
from encoders.memetic_pso import encode_memetic_pso
from encoders.memetic_ppso import encode_memetic_ppso


# 註冊所有 encoder (name -> function)
# 新方法只要在這裡加一行 + 建立對應的 configs/<name>.yml
# 'full_search':  encode_full_search,
METHODS = {
    'pso':          encode_pso,
    'ppso':         encode_ppso,
    'memetic_pso':  encode_memetic_pso,
    'memetic_ppso': encode_memetic_ppso,
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
    parser.add_argument('--output-dir', type=str, default=None,
                        help='覆寫 global.yml 的 output_dir')
    parser.add_argument('--global-config', type=str, default=None,
                        help='全域設定檔路徑 (預設 configs/global.yml)')
    parser.add_argument('--configs-dir', type=str, default=CONFIGS_DIR)
    parser.add_argument('--n-runs', type=int, default=None,
                        help='每個方法×影像跑幾次 (覆寫 global.yml)')
    args = parser.parse_args()

    # 讀取全域設定
    global_cfg = load_global_config(args.global_config)

    # CLI 參數覆寫 global config (CLI > YAML > 程式預設)
    if args.image_size    is not None: global_cfg['image_size']         = args.image_size
    if args.range_size    is not None: global_cfg['range_size']         = args.range_size
    if args.domain_size   is not None: global_cfg['domain_size']        = args.domain_size
    if args.domain_stride is not None: global_cfg['domain_stride']      = args.domain_stride
    if args.decode_iter   is not None: global_cfg['decode_iterations']  = args.decode_iter

    output_dir = args.output_dir or global_cfg.get('output_dir', 'results')
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


if __name__ == "__main__":
    main()
