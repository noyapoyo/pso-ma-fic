#!/usr/bin/env python3
"""
=============================================================================
run_comparison.py - Three-Way FGDS Comparison Experiment
=============================================================================

比較三種 Feature-Guided Domain Selection 的變體（全部使用 k=80）:
  1. fgds_pairwise  : Brute-force pairwise candidate search (無 KDTree)
  2. fgds_kdtree    : KDTree 加速 candidate search，等權重 feature
  3. fgds_ga        : KDTree 加速 candidate search，GA 優化 feature weights

時間記錄（逐項分開）:
  - feature_extract_time_sec    : feature 提取耗時
  - candidate_selection_time_sec: candidate pool 建構耗時 (pairwise 或 KDTree)
  - preprocessing_time_sec      : feature_extract + candidate_selection
  - encoding_time_sec           : 搜尋迴圈耗時
  - total_time_sec              : preprocessing + encoding
  - ga_training_time_sec        : (僅 fgds_ga) GA 訓練耗時

用法:
    # 完整執行（含 GA 訓練）
    python run_comparison.py \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 500 --top-k 80

    # 指定訓練圖片
    python run_comparison.py \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 500 --top-k 80 --train-image 0043

    # 跳過 GA 訓練，載入已有的 weights
    python run_comparison.py \\
        --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4 \\
        --ffe-budget 500 --top-k 80 \\
        --load-ga-weights results_feature_weight_ga/ga_optimization_result.json

Output:
    results_comparison/
        experiment_results_TIMESTAMP.csv    詳細逐圖時間+品質數據
        convergence_IMAGE_run0.json         三個方法的收斂曲線
        convergence_IMAGE.png               逐圖收斂圖
        convergence_average.png             平均收斂圖（所有圖平均）
        ga_weights_k{K}.json               本次 GA 訓練結果

畫圖:
    python plot_convergence.py results_comparison/
=============================================================================
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

import fic_core as core
from encoders.feature_exhaustive import encode_feature_exhaustive
from encoders.feature_kdtree import encode_feature_kdtree, build_candidate_pools_kdtree
from encoders.feature_guided_pso import extract_features
from convergence import save_convergence


# =============================================================================
# GA Training (adapted from run_feature_weight_ga.py)
# =============================================================================

def evaluate_weights_fast(weights, range_blocks, all_iso,
                           range_feats_n_base, domain_feats_n_base,
                           top_k, sample_indices):
    """
    GA fitness: weighted feature KNN + exhaustive search on sampled blocks.
    range_feats_n_base / domain_feats_n_base are already z-score normalized
    (but NOT yet scaled by weights).
    """
    from encoders.feature_kdtree import build_candidate_pools_kdtree
    w = np.array(weights, dtype=np.float64)
    rf_w = range_feats_n_base * w
    df_w = domain_feats_n_base * w
    candidate_pools = build_candidate_pools_kdtree(rf_w, df_w, top_k)

    total_mse = 0.0
    for r_idx in sample_indices:
        r_block = range_blocks[r_idx]
        pool = candidate_pools[r_idx]
        best_mse = np.inf
        for ci in range(len(pool)):
            d_idx = int(pool[ci])
            for iso in range(8):
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                if abs(s) < 1.0 and mse < best_mse:
                    best_mse = mse
        total_mse += best_mse
    return total_mse / len(sample_indices)


def run_ga_training(image, range_size, domain_size, domain_stride,
                    top_k, pop_size, n_gen, sample_ratio, seed):
    """
    Run GA to optimize feature weights. Returns (best_weights, history, precomp_time).

    Returns:
        best_weights : np.ndarray (8,)
        history      : list of {gen, best_mse, mean_mse, best_weights}
        precomp_time : float (one-time precompute cost in seconds)
    """
    print(f"\n  GA Training Parameters:")
    print(f"    Population: {pop_size}  Generations: {n_gen}  "
          f"top_k: {top_k}  sample_ratio: {sample_ratio}  seed: {seed}")

    rng = np.random.RandomState(seed)
    n_dims = 8

    # One-time precompute
    print("  Precomputing blocks, isometries, features...")
    t_pre = time.time()
    range_blocks, _ = core.extract_range_blocks(image, range_size)
    domain_blocks, _ = core.extract_domain_blocks(image, domain_size, domain_stride, range_size)
    n_range = len(range_blocks)
    all_iso = core.precompute_all_isometries(domain_blocks)
    range_feats_raw = extract_features(range_blocks)
    domain_feats_raw = extract_features(domain_blocks)
    mu = domain_feats_raw.mean(axis=0)
    sigma = domain_feats_raw.std(axis=0) + 1e-8
    range_feats_n = (range_feats_raw - mu) / sigma
    domain_feats_n = (domain_feats_raw - mu) / sigma
    precomp_time = time.time() - t_pre

    n_sample = max(1, int(n_range * sample_ratio))
    sample_indices = rng.choice(n_range, size=n_sample, replace=False)
    sample_indices.sort()
    print(f"  Done ({precomp_time:.1f}s). n_range={n_range}, n_sample={n_sample}")

    eval_kwargs = dict(
        range_blocks=range_blocks, all_iso=all_iso,
        range_feats_n_base=range_feats_n, domain_feats_n_base=domain_feats_n,
        top_k=top_k, sample_indices=sample_indices,
    )

    # Initialize population (individual 0 = equal weights baseline)
    population = np.ones((pop_size, n_dims))
    for i in range(1, pop_size):
        population[i] = rng.uniform(0.3, 2.5, size=n_dims)

    # Evaluate gen 0
    fitness = np.zeros(pop_size)
    print(f"\n  Gen 0/{n_gen} (evaluating {pop_size} individuals × {n_sample} blocks)")
    for i in range(pop_size):
        mse = evaluate_weights_fast(population[i], **eval_kwargs)
        fitness[i] = -mse

    best_idx = fitness.argmax()
    best_fitness = fitness[best_idx]
    best_weights = population[best_idx].copy()
    history = [{'gen': 0, 'best_mse': round(-best_fitness, 4),
                 'mean_mse': round(-fitness.mean(), 4),
                 'best_weights': best_weights.round(4).tolist()}]

    crossover_rate = 0.8
    mutation_rate = 0.4
    mutation_scale = 0.25
    elite_count = 2

    for gen in range(1, n_gen + 1):
        t_gen = time.time()
        new_pop = np.zeros_like(population)
        elite_idx = np.argsort(fitness)[-elite_count:]
        for i, idx in enumerate(elite_idx):
            new_pop[i] = population[idx]

        for i in range(elite_count, pop_size):
            t1 = rng.choice(pop_size, size=3, replace=False)
            p1 = population[t1[fitness[t1].argmax()]]
            t2 = rng.choice(pop_size, size=3, replace=False)
            p2 = population[t2[fitness[t2].argmax()]]
            if rng.random() < crossover_rate:
                alpha = 0.5
                lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
                hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
                child = rng.uniform(lo, hi)
            else:
                child = p1.copy()
            if rng.random() < mutation_rate:
                n_mutate = rng.randint(1, 4)
                dims = rng.choice(n_dims, size=n_mutate, replace=False)
                child[dims] += rng.normal(0, mutation_scale, size=n_mutate)
            new_pop[i] = np.maximum(child, 0.05)

        population = new_pop
        for i in range(pop_size):
            mse = evaluate_weights_fast(population[i], **eval_kwargs)
            fitness[i] = -mse

        gen_best_idx = fitness.argmax()
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_weights = population[gen_best_idx].copy()
            marker = " ★"
        else:
            marker = ""
        print(f"  Gen {gen:>2}/{n_gen}  best_mse={-best_fitness:.4f}  "
              f"w={np.round(best_weights, 2)}  ({time.time()-t_gen:.1f}s){marker}")
        history.append({'gen': gen, 'best_mse': round(-best_fitness, 4),
                        'mean_mse': round(-fitness.mean(), 4),
                        'best_weights': best_weights.round(4).tolist()})

    feature_names = ['mean', 'std', 'grad_h', 'grad_v', 'q1', 'q2', 'q3', 'q4']
    print(f"\n  GA complete. Best weights:")
    for name, w in zip(feature_names, best_weights):
        print(f"    {name:<8} {w:.4f}")
    return best_weights, history, precomp_time


# =============================================================================
# Image collection
# =============================================================================

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


# =============================================================================
# Timing extraction helpers
# =============================================================================

def extract_timing(stats, method_key):
    """
    從 encoder stats 中標準化提取計時欄位。
    回傳 dict 包含 feature_extract_time_sec, candidate_selection_time_sec,
    preprocessing_time_sec, encoding_time_sec, total_time_sec。
    """
    feat_t = stats.get('feature_extract_time_sec', 0.0)
    if method_key == 'fgds_pairwise':
        cand_t = stats.get('pairwise_pool_time_sec', 0.0)
    else:
        cand_t = stats.get('kdtree_build_query_time_sec', 0.0)
    preproc_t = stats.get('feature_setup_time_sec', feat_t + cand_t)
    enc_t = stats.get('encoding_time_sec', 0.0)
    return {
        'feature_extract_time_sec': round(feat_t, 3),
        'candidate_selection_time_sec': round(cand_t, 3),
        'preprocessing_time_sec': round(preproc_t, 3),
        'encoding_time_sec': round(enc_t, 3),
        'total_time_sec': round(preproc_t + enc_t, 3),
    }


# =============================================================================
# Summary table & CSV
# =============================================================================

def print_summary(all_results):
    print("\n" + "=" * 120)
    print("  COMPARISON RESULTS SUMMARY")
    print("=" * 120)
    header = (f"  {'Image':<10} {'Method':<22} {'PSNR':>8} {'MSE':>10} "
              f"{'FeatExtr':>10} {'CandSel':>10} {'Preproc':>10} "
              f"{'Encode':>9} {'Total':>9} {'Evals':>10}")
    print(header)
    print("-" * 120)
    for r in all_results:
        print(f"  {r['image']:<10} {r['method']:<22} "
              f"{r['psnr_actual']:>8.2f} {r['mse_mean']:>10.4f} "
              f"{r['feature_extract_time_sec']:>10.2f} "
              f"{r['candidate_selection_time_sec']:>10.2f} "
              f"{r['preprocessing_time_sec']:>10.2f} "
              f"{r['encoding_time_sec']:>9.2f} "
              f"{r['total_time_sec']:>9.2f} "
              f"{r.get('n_evaluations', 0):>10,}")
    print("=" * 120)

    # Per-method averages
    print("\n  AVERAGES ACROSS ALL IMAGES:")
    print(f"  {'Method':<22} {'PSNR':>8} {'MSE':>10} {'FeatExtr':>10} "
          f"{'CandSel':>10} {'Preproc':>10} {'Encode':>9} {'Total':>9}")
    print("-" * 90)
    for method in ['fgds_pairwise', 'fgds_kdtree', 'fgds_ga']:
        subset = [r for r in all_results if r['method'] == method]
        if not subset:
            continue
        avg = lambda key: np.mean([r[key] for r in subset])
        print(f"  {method:<22} "
              f"{avg('psnr_actual'):>8.2f} {avg('mse_mean'):>10.4f} "
              f"{avg('feature_extract_time_sec'):>10.2f} "
              f"{avg('candidate_selection_time_sec'):>10.2f} "
              f"{avg('preprocessing_time_sec'):>10.2f} "
              f"{avg('encoding_time_sec'):>9.2f} "
              f"{avg('total_time_sec'):>9.2f}")
    print("=" * 120)


def save_csv(all_results, output_dir, ga_training_time_sec):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"experiment_results_{timestamp}.csv")

    fieldnames = [
        'image', 'method',
        'psnr_actual', 'mse_mean', 'mse_max', 'compression_ratio',
        'feature_extract_time_sec', 'candidate_selection_time_sec',
        'preprocessing_time_sec', 'encoding_time_sec', 'total_time_sec',
        'ga_training_time_sec',
        'n_evaluations', 'n_range', 'n_domain',
        'fgds_top_k', 'feature_weights',
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            # GA training time only applies to fgds_ga
            if r.get('method') == 'fgds_ga':
                row['ga_training_time_sec'] = ga_training_time_sec
            if isinstance(row.get('feature_weights'), list):
                row['feature_weights'] = str(row['feature_weights'])
            writer.writerow(row)

    print(f"\n  Saved CSV: {path}")
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-way FGDS comparison: pairwise vs KDTree vs GA weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # FIC parameters
    parser.add_argument('--image-size',    type=int, default=1024)
    parser.add_argument('--range-size',    type=int, default=4)
    parser.add_argument('--domain-size',   type=int, default=8)
    parser.add_argument('--domain-stride', type=int, default=4)
    parser.add_argument('--ffe-budget',    type=int, default=500)
    parser.add_argument('--top-k',         type=int, default=80)
    # Image
    parser.add_argument('--image-dir',  type=str, default='images')
    parser.add_argument('--image',      type=str, default=None,
                        help='Single image name (default: all in image-dir)')
    # Output
    parser.add_argument('--output-dir', type=str, default='results_comparison')
    # GA
    parser.add_argument('--train-image',     type=str, default='0043',
                        help='Image used for GA training')
    parser.add_argument('--ga-pop',          type=int, default=16)
    parser.add_argument('--ga-gen',          type=int, default=10)
    parser.add_argument('--ga-sample-ratio', type=float, default=0.02,
                        help='Fraction of range blocks sampled per GA fitness eval')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--load-ga-weights', type=str, default=None,
                        help='Skip GA training; load weights from this JSON file')
    args = parser.parse_args()

    print(f"\n{'═' * 72}")
    print(f"  Three-Way FGDS Comparison Experiment")
    print(f"{'═' * 72}")
    print(f"  image_size={args.image_size}  range={args.range_size}  "
          f"domain={args.domain_size}  stride={args.domain_stride}")
    print(f"  ffe_budget={args.ffe_budget}  top_k={args.top_k}")
    print(f"  output_dir={args.output_dir}/")

    os.makedirs(args.output_dir, exist_ok=True)
    images = collect_images(args.image_dir, args.image)
    print(f"  Images: {[os.path.basename(p) for p in images]}")

    # =========================================================================
    # Phase 1: GA Training
    # =========================================================================
    print(f"\n{'─' * 72}")
    print(f"  Phase 1: GA Feature Weight Optimization (k={args.top_k})")
    print(f"{'─' * 72}")

    ga_training_time_sec = 0.0

    if args.load_ga_weights:
        print(f"  Loading pre-trained weights from {args.load_ga_weights}")
        with open(args.load_ga_weights) as f:
            ga_data = json.load(f)
        ga_weights = np.array(ga_data['best_weights'])
        print(f"  Loaded weights: {np.round(ga_weights, 4)}")
        print(f"  (Training time not recorded for pre-loaded weights)")
    else:
        # Find training image
        train_path = None
        for ext in ['', '.png', '.jpg', '.bmp']:
            cand = os.path.join(args.image_dir, args.train_image + ext)
            if os.path.exists(cand):
                train_path = cand
                break
        if train_path is None:
            print(f"Error: training image '{args.train_image}' not found.")
            sys.exit(1)
        print(f"  Training image: {train_path}")
        train_image = core.load_image_as_gray(train_path, args.image_size)

        t_ga_start = time.time()
        ga_weights, ga_history, _ = run_ga_training(
            train_image,
            range_size=args.range_size,
            domain_size=args.domain_size,
            domain_stride=args.domain_stride,
            top_k=args.top_k,
            pop_size=args.ga_pop,
            n_gen=args.ga_gen,
            sample_ratio=args.ga_sample_ratio,
            seed=args.seed,
        )
        ga_training_time_sec = round(time.time() - t_ga_start, 2)
        print(f"\n  GA training time: {ga_training_time_sec:.2f}s")

        # Save GA weights
        ga_out = {
            'best_weights': ga_weights.tolist(),
            'best_mse': ga_history[-1]['best_mse'],
            'train_image': args.train_image,
            'top_k': args.top_k,
            'ga_params': {'pop_size': args.ga_pop, 'n_gen': args.ga_gen,
                          'sample_ratio': args.ga_sample_ratio, 'seed': args.seed},
            'history': ga_history,
        }
        ga_json_path = os.path.join(args.output_dir, f"ga_weights_k{args.top_k}.json")
        with open(ga_json_path, 'w') as f:
            json.dump(ga_out, f, indent=2)
        print(f"  Saved GA weights: {ga_json_path}")

    # Common encoder kwargs
    common_kwargs = dict(
        image_size=args.image_size,
        range_size=args.range_size,
        domain_size=args.domain_size,
        domain_stride=args.domain_stride,
        decode_iterations=20,
        save_fic=False,
    )
    fic_encoder_kwargs = dict(
        top_k=args.top_k,
        feature_normalize=True,
        ffe_budget_per_block=args.ffe_budget,
    )

    # =========================================================================
    # Phase 2: Run experiments on each image
    # =========================================================================
    print(f"\n{'─' * 72}")
    print(f"  Phase 2: Encoding Experiments")
    print(f"{'─' * 72}")

    all_results = []
    conv_groups = defaultdict(dict)  # image_name → {method: curve}

    for image_path in images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'━' * 72}")
        print(f"  Image: {image_name}")
        print(f"{'━' * 72}")

        # --- Method 1: FGDS Pairwise (no KDTree) ---
        print(f"\n  [1/3] fgds_pairwise  (brute-force candidate search, k={args.top_k})")
        stats_p = core.run_pipeline(
            encode_feature_exhaustive, image_path,
            method_name='fgds_pairwise',
            output_dir=args.output_dir,
            save_outputs=True,
            **common_kwargs,
            **fic_encoder_kwargs,
        )
        timing_p = extract_timing(stats_p, 'fgds_pairwise')
        stats_p.update(timing_p)
        all_results.append(stats_p)
        if stats_p.get('convergence_curve'):
            conv_groups[image_name]['fgds_pairwise'] = stats_p['convergence_curve']

        # --- Method 2: FGDS KDTree (equal weights) ---
        print(f"\n  [2/3] fgds_kdtree  (KDTree candidate search, equal weights, k={args.top_k})")
        stats_k = core.run_pipeline(
            encode_feature_kdtree, image_path,
            method_name='fgds_kdtree',
            output_dir=args.output_dir,
            save_outputs=True,
            feature_weights=None,
            **common_kwargs,
            **fic_encoder_kwargs,
        )
        timing_k = extract_timing(stats_k, 'fgds_kdtree')
        stats_k.update(timing_k)
        all_results.append(stats_k)
        if stats_k.get('convergence_curve'):
            conv_groups[image_name]['fgds_kdtree'] = stats_k['convergence_curve']

        # --- Method 3: FGDS KDTree + GA weights ---
        print(f"\n  [3/3] fgds_ga  (KDTree + GA-optimized weights, k={args.top_k})")
        stats_g = core.run_pipeline(
            encode_feature_kdtree, image_path,
            method_name='fgds_ga',
            output_dir=args.output_dir,
            save_outputs=True,
            feature_weights=ga_weights.tolist(),
            **common_kwargs,
            **fic_encoder_kwargs,
        )
        timing_g = extract_timing(stats_g, 'fgds_ga')
        stats_g.update(timing_g)
        stats_g['ga_training_time_sec'] = ga_training_time_sec
        all_results.append(stats_g)
        if stats_g.get('convergence_curve'):
            conv_groups[image_name]['fgds_ga'] = stats_g['convergence_curve']

        # Save convergence JSON for this image (all 3 methods together)
        if conv_groups[image_name]:
            conv_path = save_convergence(conv_groups[image_name],
                                         args.output_dir, image_name, run_idx=0)
            print(f"\n  Saved convergence: {conv_path}")

    # =========================================================================
    # Phase 3: Summary & Save
    # =========================================================================
    print_summary(all_results)
    save_csv(all_results, args.output_dir, ga_training_time_sec)

    print(f"\n  GA training time: {ga_training_time_sec:.2f}s  "
          f"(applies to fgds_ga method only)")

    print(f"\n{'═' * 72}")
    print(f"  All done! Results in {args.output_dir}/")
    print(f"{'═' * 72}")
    print(f"\n  To plot convergence curves:")
    print(f"    python plot_convergence.py {args.output_dir}/")
    print(f"  To plot a single image:")
    print(f"    python plot_convergence.py {args.output_dir}/convergence_IMAGE_run0.json")


if __name__ == "__main__":
    main()
