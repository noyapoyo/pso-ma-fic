"""
=============================================================================
run_random_baseline.py - Experiment 4: Random-K vs FGDS-K Exhaustive Baseline
=============================================================================

■ 研究問題:
  Feature-Guided Domain Selection (FGDS) 的優勢來自「feature-guided 候選池的品質」，
  還是只是「縮小搜索空間（做 exhaustive）」本身就有效？

■ 實驗設計:
  - Random-80 exhaustive : 隨機選 80 個 domain blocks，做窮舉搜索
  - FGDS-80 exhaustive   : KDTree feature-guided 選 80 個，做窮舉搜索

  兩者 FFE budget 完全相同（= 80 × 8 = 640 < 800）。
  唯一差異是候選池的「選取方式」。

■ 解讀:
  - FGDS-80 >> Random-80 → feature-guided pool 有實質價值
  - FGDS-80 ≈ Random-80  → 縮小搜索空間本身才是關鍵（feature 貢獻有限）

用法:
    python run_random_baseline.py \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 800 --top-k 80 --output-dir results_exp4

    # 快速測試單張圖
    python run_random_baseline.py --image 0007 \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 800 --top-k 80 --output-dir results_exp4_test

畫圖:
    python plot_convergence.py results_exp4/

結果: results_exp4/
    - experiment_results_TIMESTAMP.csv
    - convergence_IMAGE_run0.json  (random_k80 + fgds_k80 for each image)
=============================================================================
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

import fic_core as core
from encoders.random_exhaustive import encode_random_exhaustive
from encoders.feature_kdtree import encode_feature_kdtree
from convergence import save_convergence


def collect_images(image_dir, single_image=None):
    if single_image:
        for ext in ['', '.png', '.jpg', '.bmp']:
            cand = os.path.join(image_dir, single_image + ext)
            if os.path.exists(cand):
                return [cand]
        if os.path.exists(single_image):
            return [single_image]
        print(f"Error: image '{single_image}' not found.")
        sys.exit(1)
    files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    if not files:
        print(f"No images in {image_dir}/")
        sys.exit(1)
    return files


def print_summary(all_results):
    print("\n" + "=" * 100)
    print("  RANDOM vs FGDS BASELINE RESULTS SUMMARY")
    print("=" * 100)
    header = (f"  {'Image':<10} {'Method':<22} {'PSNR(dB)':>10} "
              f"{'AvgMSE':>12} {'Encode(s)':>11} {'Evals':>12}")
    print(header)
    print("-" * 100)
    for r in all_results:
        print(f"  {r['image']:<10} {r['method']:<22} "
              f"{r['psnr_actual']:>10.2f} {r['mse_mean']:>12.4f} "
              f"{r['encoding_time_sec']:>11.2f} "
              f"{r.get('n_evaluations', 0):>12,}")
    print("=" * 100)

    print("\n  AVERAGES BY METHOD:")
    print(f"  {'Method':<22} {'Avg PSNR':>10} {'Avg MSE':>12} {'Avg Encode(s)':>14}")
    print("-" * 65)
    for method in ['random_k80', 'fgds_k80']:
        subset = [r for r in all_results if r['method'] == method]
        if not subset:
            continue
        avg_psnr = np.mean([r['psnr_actual'] for r in subset])
        avg_mse = np.mean([r['mse_mean'] for r in subset])
        avg_enc = np.mean([r['encoding_time_sec'] for r in subset])
        print(f"  {method:<22} {avg_psnr:>10.2f} {avg_mse:>12.4f} {avg_enc:>14.2f}")
    print("=" * 100)

    # PSNR gap
    r80 = [r for r in all_results if r['method'] == 'random_k80']
    f80 = [r for r in all_results if r['method'] == 'fgds_k80']
    if r80 and f80:
        gap = np.mean([r['psnr_actual'] for r in f80]) - np.mean([r['psnr_actual'] for r in r80])
        print(f"\n  FGDS-80 vs Random-80 PSNR gap: {gap:+.2f} dB "
              f"({'FGDS wins' if gap > 0 else 'Random comparable'})")


def save_csv(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"experiment_results_{timestamp}.csv")

    fieldnames = [
        'image', 'method',
        'psnr_actual', 'mse_mean', 'mse_max', 'compression_ratio',
        'encoding_time_sec', 'feature_setup_time_sec',
        'n_evaluations', 'n_range', 'n_domain',
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\n  Saved CSV: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Random-K vs FGDS-K Exhaustive Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image-size',    type=int, default=1024)
    parser.add_argument('--range-size',    type=int, default=4)
    parser.add_argument('--domain-size',   type=int, default=8)
    parser.add_argument('--domain-stride', type=int, default=4)
    parser.add_argument('--ffe-budget',    type=int, default=800)
    parser.add_argument('--top-k',         type=int, default=80)
    parser.add_argument('--image-dir',     type=str, default='images')
    parser.add_argument('--image',         type=str, default=None,
                        help='Single image name (default: all in image-dir)')
    parser.add_argument('--output-dir',    type=str, default='results_exp4')
    parser.add_argument('--seed',          type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images = collect_images(args.image_dir, args.image)

    print(f"\n{'═' * 72}")
    print(f"  Experiment 4: Random-{args.top_k} vs FGDS-{args.top_k} Exhaustive")
    print(f"{'═' * 72}")
    print(f"  image_size={args.image_size}  range={args.range_size}  "
          f"domain={args.domain_size}  stride={args.domain_stride}")
    print(f"  ffe_budget={args.ffe_budget}  top_k={args.top_k}  seed={args.seed}")
    print(f"  Images: {[os.path.basename(p) for p in images]}")
    print(f"  Output: {args.output_dir}/")

    common_kwargs = dict(
        image_size=args.image_size,
        range_size=args.range_size,
        domain_size=args.domain_size,
        domain_stride=args.domain_stride,
        decode_iterations=20,
        save_fic=False,
        top_k=args.top_k,
        ffe_budget_per_block=args.ffe_budget,
    )

    all_results = []
    conv_groups = defaultdict(dict)  # image_name → {method: curve}

    for image_path in images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'━' * 72}")
        print(f"  Image: {image_name}")
        print(f"{'━' * 72}")

        # --- Method 1: Random-K Exhaustive ---
        print(f"\n  [1/2] random_k{args.top_k}  (random domain pool, exhaustive)")
        stats_r = core.run_pipeline(
            encode_random_exhaustive, image_path,
            method_name=f'random_k{args.top_k}',
            output_dir=args.output_dir,
            save_outputs=True,
            seed=args.seed,
            **common_kwargs,
        )
        all_results.append(stats_r)
        if stats_r.get('convergence_curve'):
            conv_groups[image_name][f'random_k{args.top_k}'] = stats_r['convergence_curve']

        # --- Method 2: FGDS-K Exhaustive (KDTree feature-guided) ---
        print(f"\n  [2/2] fgds_k{args.top_k}  (KDTree feature-guided pool, exhaustive)")
        stats_f = core.run_pipeline(
            encode_feature_kdtree, image_path,
            method_name=f'fgds_k{args.top_k}',
            output_dir=args.output_dir,
            save_outputs=True,
            feature_weights=None,
            feature_normalize=True,
            **common_kwargs,
        )
        all_results.append(stats_f)
        if stats_f.get('convergence_curve'):
            conv_groups[image_name][f'fgds_k{args.top_k}'] = stats_f['convergence_curve']

        # Save convergence JSON for this image
        if conv_groups[image_name]:
            conv_path = save_convergence(
                conv_groups[image_name], args.output_dir, image_name, run_idx=0
            )
            print(f"\n  Saved convergence: {conv_path}")

    print_summary(all_results)
    save_csv(all_results, args.output_dir)

    print(f"\n{'═' * 72}")
    print(f"  All done! Results in {args.output_dir}/")
    print(f"{'═' * 72}")
    print(f"\n  To plot convergence curves:")
    print(f"    python plot_convergence.py {args.output_dir}/")


if __name__ == "__main__":
    main()
