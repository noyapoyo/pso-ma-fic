"""
=============================================================================
run_k_sensitivity.py - Experiment 3: FGDS Candidate Pool Size (K) Sensitivity
=============================================================================

對 FGDS+KDTree 方法測試不同的 candidate pool size K，
觀察 K 如何影響 PSNR 和 encoding time。

■ K 值: 5, 10, 20, 40, 80, 160
■ 每個 K 值在全部 10 張圖各跑一次 (deterministic, feature pool 不含隨機)
■ 使用 FGDS+KDTree（快速前處理），FFE budget=800 統一 x 軸比較

結論: K=80 在品質和速度之間取得最好平衡。

用法:
    # 完整執行（10 張圖 × 6 個 K 值）
    python run_k_sensitivity.py \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 800 --k-values 5 10 20 40 80 160 --output-dir results_exp3

    # 快速測試單張圖
    python run_k_sensitivity.py --image 0007 \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 800 --output-dir results_exp3_test

畫圖:
    python plot_convergence.py results_exp3/

結果: results_exp3/
    - k_sensitivity_TIMESTAMP.csv
    - convergence_IMAGE_run0.json  (all K values for each image)
=============================================================================
"""

import argparse
import os
import sys
import numpy as np
import csv
from datetime import datetime

import fic_core as core
from encoders.feature_kdtree import encode_feature_kdtree


def collect_images(image_dir, single_image=None):
    """收集影像清單。"""
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
        print(f"No images found in {image_dir}/")
        sys.exit(1)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="FGDS K-Sensitivity Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--k-values', nargs='+', type=int,
                        default=[5, 10, 20, 40, 80, 160],
                        help='K values to test')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image name (default: all in images/)')
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--range-size', type=int, default=4)
    parser.add_argument('--domain-size', type=int, default=8)
    parser.add_argument('--domain-stride', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='results_exp3')
    parser.add_argument('--ffe-budget', type=int, default=800,
                        help='FFE budget per block (must match Exp1 for fair comparison; '
                             'K=80 uses 640 FFE, budget=800 leaves headroom)')
    args = parser.parse_args()

    images = collect_images(args.image_dir, args.image)

    print(f"\n{'=' * 72}")
    print(f"  FGDS K-Sensitivity Experiment")
    print(f"{'=' * 72}")
    print(f"  K values:     {args.k_values}")
    print(f"  Images:       {[os.path.basename(p) for p in images]}")
    print(f"  Image size:   {args.image_size}×{args.image_size}")
    print(f"  Block params: range={args.range_size}, "
          f"domain={args.domain_size}, stride={args.domain_stride}")
    print(f"  Output:       {args.output_dir}/")

    # 預估時間
    n_configs = len(args.k_values) * len(images)
    est_minutes = n_configs * 6  # ~6 min per (K, image) on 1024
    print(f"  Configs:      {n_configs} total ({len(args.k_values)} K × {len(images)} images)")
    print(f"  Est. time:    ~{est_minutes} min ({est_minutes/60:.1f} hr)")
    print(f"{'=' * 72}\n")

    all_results = []

    for k in args.k_values:
        for image_path in images:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            method_name = f"fgds_k{k}"

            print(f"\n{'─' * 60}")
            print(f"  K={k}  image={image_name}  "
                  f"FFEs/block={k * 8}")
            print(f"{'─' * 60}")

            stats = core.run_pipeline(
                encode_feature_kdtree,
                image_path,
                method_name=method_name,
                output_dir=args.output_dir,
                image_size=args.image_size,
                range_size=args.range_size,
                domain_size=args.domain_size,
                domain_stride=args.domain_stride,
                decode_iterations=20,
                save_fic=False,
                save_outputs=(k == 80),  # 只存 K=80 的 reconstructed image
                # encoder kwargs:
                top_k=k,
                feature_normalize=True,
                ffe_budget_per_block=args.ffe_budget,  # 統一 budget 以便與 PSO 比較
            )
            stats['top_k'] = k
            stats['ffe_per_block'] = k * 8
            all_results.append(stats)

    # ============================================================
    # Print summary table
    # ============================================================
    print(f"\n\n{'=' * 110}")
    print(f"  K-SENSITIVITY RESULTS SUMMARY")
    print(f"{'=' * 110}")
    print(f"  {'K':>5}  {'FFE':>6}  {'Image':<10}  {'Preproc(s)':>12}  "
          f"{'Encode(s)':>11}  {'Total(s)':>10}  {'PSNR(dB)':>10}  {'AvgMSE':>12}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*12}  "
          f"{'-'*11}  {'-'*10}  {'-'*10}  {'-'*12}")

    for r in sorted(all_results, key=lambda x: (x['image'], x['top_k'])):
        preproc = r.get('feature_setup_time_sec', 0)
        encode = r['encoding_time_sec']
        total = preproc + encode
        print(f"  {r['top_k']:>5}  {r['ffe_per_block']:>6}  {r['image']:<10}  "
              f"{preproc:>12.2f}  {encode:>11.2f}  {total:>10.2f}  "
              f"{r['psnr_actual']:>10.2f}  {r['mse_mean']:>12.4f}")

    # Per-K average
    print(f"\n  {'─' * 80}")
    print(f"  {'K':>5}  {'FFE':>6}  {'Avg Preproc':>12}  {'Avg Encode':>11}  "
          f"{'Avg Total':>10}  {'Avg PSNR':>10}  {'Avg MSE':>12}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*12}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*12}")

    from collections import defaultdict
    by_k = defaultdict(list)
    for r in all_results:
        by_k[r['top_k']].append(r)

    for k in sorted(by_k.keys()):
        runs = by_k[k]
        avg_pre = np.mean([r.get('feature_setup_time_sec', 0) for r in runs])
        avg_enc = np.mean([r['encoding_time_sec'] for r in runs])
        avg_psnr = np.mean([r['psnr_actual'] for r in runs])
        avg_mse = np.mean([r['mse_mean'] for r in runs])
        print(f"  {k:>5}  {k*8:>6}  {avg_pre:>12.2f}  {avg_enc:>11.2f}  "
              f"{avg_pre+avg_enc:>10.2f}  {avg_psnr:>10.2f}  {avg_mse:>12.4f}")

    print(f"{'=' * 110}")

    # ============================================================
    # Save CSV
    # ============================================================
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"k_sensitivity_{timestamp}.csv")

    fieldnames = ['image', 'method', 'top_k', 'ffe_per_block',
                  'encoding_time_sec', 'feature_setup_time_sec',
                  'feature_extract_time_sec', 'kdtree_build_query_time_sec',
                  'psnr_actual', 'psnr_db', 'mse_mean', 'mse_max',
                  'n_evaluations', 'n_range', 'n_domain',
                  'compression_ratio', 'bits_per_code']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\n  Saved CSV: {csv_path}")

    # ============================================================
    # Save convergence data (compatible with plot_convergence.py)
    # ============================================================
    from convergence import save_convergence
    from collections import defaultdict as _dd

    conv_groups = _dd(dict)  # (image,) → {method: curve}
    for r in all_results:
        curve = r.get('convergence_curve')
        if curve:
            conv_groups[r['image']][r['method']] = curve

    for image_name, curves in conv_groups.items():
        path = save_convergence(curves, args.output_dir, image_name, run_idx=0)
        print(f"  Saved convergence: {path}")


if __name__ == "__main__":
    main()
