"""
=============================================================================
run_feature_weight_ga.py - GA-based Feature Weight Optimization for FGDS
=============================================================================

用 Genetic Algorithm 來尋找最佳的 feature weight vector w ∈ R^8。

■ 背景:
  FGDS 使用 8 維 isometry-invariant feature 做 KNN domain selection。
  目前 z-score normalize 後等權重計算 Euclidean distance，但不同 feature
  dimension 對 MSE matching 的重要性不同。例如 mean 直接影響 brightness offset o,
  std 直接影響 contrast s, 而 gradient 可能相對不重要。

■ 做法:
  1. 用 GA 在一張 training image 的 sample 上最小化 avg MSE
     - 每個 individual = weight vector w ∈ R^8
     - fitness = -avg_MSE (用 weighted feature 做 top-K selection 後 exhaustive search)
  2. 將最佳 weights 應用到所有 5 張圖, 與 equal-weight baseline 比較

■ 為什麼這是 heuristic:
  GA 是 metaheuristic，用在了 feature space 的設計上 (weight optimization)。
  這和用 PSO 做 domain matching 不同：GA 優化的是 preprocessing 的品質，
  而非 per-block 的搜尋過程。

用法:
    # 在 0043 上做 GA 優化, 然後對所有圖做 evaluation (推薦先跑這個)
    python run_feature_weight_ga.py --train-image 0043

    # 用更大的 GA 設定 (更多 generation 和 population)
    python run_feature_weight_ga.py --train-image 0043 --ga-pop 20 --ga-gen 15

    # 快速測試 (小 population, 少 generation, 多 sample)
    python run_feature_weight_ga.py --train-image 0043 --ga-pop 10 --ga-gen 5 --sample-ratio 0.01

結果儲存在 results_feature_weight_ga/:
    - ga_optimization_result.json (最佳 weights + GA 歷史)
    - feature_weight_comparison_YYYYMMDD_HHMMSS.csv
=============================================================================
"""

import argparse
import os
import sys
import numpy as np
import csv
import json
import time
from datetime import datetime

import fic_core as core
from encoders.feature_guided_pso import extract_features
from encoders.feature_kdtree import (
    encode_feature_kdtree,
    build_candidate_pools_kdtree,
)


# =============================================================================
# GA Fitness Function
# =============================================================================

def evaluate_weights(weights, image, range_blocks, domain_blocks,
                     all_iso, range_feats_raw, domain_feats_raw,
                     domain_mu, domain_sigma,
                     top_k, sample_indices):
    """
    Evaluate a weight vector on sampled range blocks.

    這是 GA 的 fitness function:
      1. 用 weights 對 z-score normalized features 做 weighted scaling
      2. 用 KD-Tree 建 candidate pool
      3. 在 sampled range blocks 上做 exhaustive search
      4. 回傳 avg MSE (越低越好)

    為了效率, feature extraction 和 normalization 只做一次 (在外部),
    這裡只做 weighting + KD-Tree + exhaustive。
    """
    w = np.array(weights, dtype=np.float64)

    # Apply weights to normalized features
    domain_feats_w = ((domain_feats_raw - domain_mu) / domain_sigma) * w
    range_feats_w = ((range_feats_raw - domain_mu) / domain_sigma) * w

    # Build KD-Tree candidate pool
    candidate_pools = build_candidate_pools_kdtree(
        range_feats_w, domain_feats_w, top_k
    )

    # Exhaustive search on sampled blocks
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


# =============================================================================
# GA Main Loop
# =============================================================================

def run_ga_optimization(image, range_size, domain_size, domain_stride,
                        top_k, pop_size, n_gen, sample_ratio, seed):
    """
    Run GA to find optimal feature weights.

    Returns:
        best_weights: np.ndarray (8,)
        history: list of dicts with generation-level stats
    """
    print(f"\n  GA Parameters:")
    print(f"    Population:    {pop_size}")
    print(f"    Generations:   {n_gen}")
    print(f"    Sample ratio:  {sample_ratio}")
    print(f"    Top-K:         {top_k}")
    print(f"    Seed:          {seed}")

    rng = np.random.RandomState(seed)
    n_dims = 8

    # --- Precompute everything that doesn't depend on weights ---
    print(f"\n  Precomputing blocks and features (one-time cost)...")
    t0 = time.time()

    range_blocks, _ = core.extract_range_blocks(image, range_size)
    domain_blocks, _ = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)

    all_iso = core.precompute_all_isometries(domain_blocks)
    range_feats_raw = extract_features(range_blocks)
    domain_feats_raw = extract_features(domain_blocks)

    domain_mu = domain_feats_raw.mean(axis=0)
    domain_sigma = domain_feats_raw.std(axis=0) + 1e-8

    # Sample range blocks for fitness evaluation
    n_sample = max(1, int(n_range * sample_ratio))
    sample_indices = rng.choice(n_range, size=n_sample, replace=False)
    sample_indices.sort()

    precomp_time = time.time() - t0
    print(f"  Done ({precomp_time:.1f}s). n_range={n_range}, n_sample={n_sample}")

    # Shared args for fitness evaluation
    eval_kwargs = dict(
        image=image, range_blocks=range_blocks, domain_blocks=domain_blocks,
        all_iso=all_iso, range_feats_raw=range_feats_raw,
        domain_feats_raw=domain_feats_raw,
        domain_mu=domain_mu, domain_sigma=domain_sigma,
        top_k=top_k, sample_indices=sample_indices,
    )

    # --- Initialize population ---
    population = np.ones((pop_size, n_dims))
    # Individual 0 = equal weights (baseline)
    for i in range(1, pop_size):
        population[i] = rng.uniform(0.3, 2.5, size=n_dims)

    # --- Evaluate initial population ---
    fitness = np.zeros(pop_size)  # -avg_mse (higher is better)
    print(f"\n  Gen 0 / {n_gen}  (evaluating {pop_size} individuals × {n_sample} blocks)")
    for i in range(pop_size):
        mse = evaluate_weights(population[i], **eval_kwargs)
        fitness[i] = -mse
        if i < 5 or i == pop_size - 1:
            print(f"    [{i:>3}] w={np.round(population[i], 2)}  mse={mse:.2f}")

    best_ever_idx = fitness.argmax()
    best_ever_fitness = fitness[best_ever_idx]
    best_ever_weights = population[best_ever_idx].copy()

    history = [{
        'gen': 0,
        'best_mse': round(-best_ever_fitness, 4),
        'mean_mse': round(-fitness.mean(), 4),
        'best_weights': best_ever_weights.round(4).tolist(),
    }]

    # --- GA Main Loop ---
    crossover_rate = 0.8
    mutation_rate = 0.4
    mutation_scale = 0.25
    elite_count = 2

    for gen in range(1, n_gen + 1):
        t_gen = time.time()

        # Elitism
        new_pop = np.zeros_like(population)
        elite_idx = np.argsort(fitness)[-elite_count:]
        for i, idx in enumerate(elite_idx):
            new_pop[i] = population[idx]

        # Generate offspring
        for i in range(elite_count, pop_size):
            # Tournament selection (size=3)
            t1 = rng.choice(pop_size, size=3, replace=False)
            p1 = population[t1[fitness[t1].argmax()]]
            t2 = rng.choice(pop_size, size=3, replace=False)
            p2 = population[t2[fitness[t2].argmax()]]

            # BLX-alpha crossover
            if rng.random() < crossover_rate:
                alpha = 0.5
                lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
                hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
                child = rng.uniform(lo, hi)
            else:
                child = p1.copy()

            # Gaussian mutation
            if rng.random() < mutation_rate:
                # Mutate 1-3 random dimensions
                n_mutate = rng.randint(1, 4)
                dims = rng.choice(n_dims, size=n_mutate, replace=False)
                child[dims] += rng.normal(0, mutation_scale, size=n_mutate)

            # Clamp to positive
            child = np.maximum(child, 0.05)
            new_pop[i] = child

        population = new_pop

        # Evaluate
        for i in range(pop_size):
            mse = evaluate_weights(population[i], **eval_kwargs)
            fitness[i] = -mse

        gen_best_idx = fitness.argmax()
        gen_best_mse = -fitness[gen_best_idx]
        gen_time = time.time() - t_gen

        if fitness[gen_best_idx] > best_ever_fitness:
            best_ever_fitness = fitness[gen_best_idx]
            best_ever_weights = population[gen_best_idx].copy()
            marker = " ★ NEW BEST"
        else:
            marker = ""

        print(f"  Gen {gen:>2} / {n_gen}  "
              f"best_mse={gen_best_mse:.2f}  "
              f"best_ever={-best_ever_fitness:.2f}  "
              f"w={np.round(population[gen_best_idx], 2)}  "
              f"({gen_time:.1f}s){marker}")

        history.append({
            'gen': gen,
            'best_mse': round(-best_ever_fitness, 4),
            'mean_mse': round(-fitness.mean(), 4),
            'best_weights': best_ever_weights.round(4).tolist(),
        })

    print(f"\n  {'═' * 60}")
    print(f"  GA COMPLETE")
    print(f"  Best weights:  {np.round(best_ever_weights, 4)}")
    print(f"  Best avg MSE:  {-best_ever_fitness:.4f}")
    feature_names = ['mean', 'std', 'grad_h', 'grad_v',
                     'q1', 'q2', 'q3', 'q4']
    print(f"  Per-feature:   " + "  ".join(
        f"{name}={w:.3f}" for name, w in zip(feature_names, best_ever_weights)))
    print(f"  {'═' * 60}")

    return best_ever_weights, history


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
        print(f"No images found in {image_dir}/")
        sys.exit(1)
    return files


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GA Feature Weight Optimization for FGDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # GA parameters
    parser.add_argument('--train-image', type=str, default='0043',
                        help='Image to use for GA training')
    parser.add_argument('--ga-pop', type=int, default=16,
                        help='GA population size')
    parser.add_argument('--ga-gen', type=int, default=10,
                        help='GA generations')
    parser.add_argument('--sample-ratio', type=float, default=0.02,
                        help='Fraction of range blocks to sample for GA fitness '
                             '(0.02 = 2%% ≈ 1310 blocks for 1024×1024)')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Candidate pool size K')
    parser.add_argument('--seed', type=int, default=42)

    # Image parameters
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--range-size', type=int, default=4)
    parser.add_argument('--domain-size', type=int, default=8)
    parser.add_argument('--domain-stride', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='results_feature_weight_ga')
    parser.add_argument('--ffe-budget', type=int, default=500,
                        help='FFE budget per block (for convergence curve x-axis)')

    # Optional: skip GA and use pre-computed weights
    parser.add_argument('--load-weights', type=str, default=None,
                        help='Path to ga_optimization_result.json (skip GA, just evaluate)')

    args = parser.parse_args()

    print(f"\n{'═' * 72}")
    print(f"  Feature Weight GA Optimization for FGDS")
    print(f"{'═' * 72}")

    # ============================================================
    # Phase 1: GA Optimization (or load pre-computed weights)
    # ============================================================
    if args.load_weights:
        print(f"  Loading pre-computed weights from {args.load_weights}")
        with open(args.load_weights) as f:
            ga_data = json.load(f)
        best_weights = np.array(ga_data['best_weights'])
        print(f"  Loaded weights: {np.round(best_weights, 4)}")
    else:
        # Load training image
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

        best_weights, history = run_ga_optimization(
            train_image,
            range_size=args.range_size,
            domain_size=args.domain_size,
            domain_stride=args.domain_stride,
            top_k=args.top_k,
            pop_size=args.ga_pop,
            n_gen=args.ga_gen,
            sample_ratio=args.sample_ratio,
            seed=args.seed,
        )

        # Save GA results
        os.makedirs(args.output_dir, exist_ok=True)
        ga_result = {
            'best_weights': best_weights.tolist(),
            'best_mse': round(float(-(-1)), 4),  # will be overwritten
            'train_image': args.train_image,
            'ga_params': {
                'pop_size': args.ga_pop,
                'n_gen': args.ga_gen,
                'sample_ratio': args.sample_ratio,
                'top_k': args.top_k,
                'seed': args.seed,
            },
            'history': history,
        }
        # Update best_mse from history
        ga_result['best_mse'] = history[-1]['best_mse']

        ga_path = os.path.join(args.output_dir, "ga_optimization_result.json")
        with open(ga_path, 'w') as f:
            json.dump(ga_result, f, indent=2)
        print(f"\n  Saved GA result: {ga_path}")

    # ============================================================
    # Phase 2: Full evaluation on all images
    # ============================================================
    print(f"\n\n{'═' * 72}")
    print(f"  Phase 2: Full Evaluation (equal weights vs GA weights)")
    print(f"{'═' * 72}\n")

    all_images = collect_images(args.image_dir)
    all_results = []

    # --- Run with equal weights (baseline) ---
    for image_path in all_images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n  [equal_weights] {image_name}")

        stats = core.run_pipeline(
            encode_feature_kdtree, image_path,
            method_name="fgds_equal_weights",
            output_dir=args.output_dir,
            image_size=args.image_size,
            range_size=args.range_size,
            domain_size=args.domain_size,
            domain_stride=args.domain_stride,
            decode_iterations=20,
            save_fic=False,
            save_outputs=False,
            top_k=args.top_k,
            feature_normalize=True,
            feature_weights=None,
            ffe_budget_per_block=args.ffe_budget,
        )
        stats['weight_type'] = 'equal'
        all_results.append(stats)

    # --- Run with GA-optimized weights ---
    for image_path in all_images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n  [ga_weights] {image_name}")

        stats = core.run_pipeline(
            encode_feature_kdtree, image_path,
            method_name="fgds_ga_weights",
            output_dir=args.output_dir,
            image_size=args.image_size,
            range_size=args.range_size,
            domain_size=args.domain_size,
            domain_stride=args.domain_stride,
            decode_iterations=20,
            save_fic=False,
            save_outputs=True,
            top_k=args.top_k,
            feature_normalize=True,
            feature_weights=best_weights.tolist(),
            ffe_budget_per_block=args.ffe_budget,
        )
        stats['weight_type'] = 'ga_optimized'
        stats['ga_weights'] = best_weights.tolist()
        all_results.append(stats)

    # ============================================================
    # Print comparison
    # ============================================================
    print(f"\n\n{'═' * 100}")
    print(f"  FEATURE WEIGHT COMPARISON (K={args.top_k})")
    print(f"{'═' * 100}")
    print(f"  {'Weight Type':<20}  {'Image':<10}  "
          f"{'PSNR(dB)':>10}  {'AvgMSE':>12}  {'Encode(s)':>11}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*11}")

    for r in sorted(all_results, key=lambda x: (x['image'], x['weight_type'])):
        print(f"  {r['weight_type']:<20}  {r['image']:<10}  "
              f"{r['psnr_actual']:>10.2f}  {r['mse_mean']:>12.4f}  "
              f"{r['encoding_time_sec']:>11.2f}")

    # Per-type average
    print(f"\n  {'─'*70}")
    for wt in ['equal', 'ga_optimized']:
        subset = [r for r in all_results if r['weight_type'] == wt]
        if subset:
            avg_psnr = np.mean([r['psnr_actual'] for r in subset])
            avg_mse = np.mean([r['mse_mean'] for r in subset])
            print(f"  {wt:<20}  {'AVERAGE':<10}  {avg_psnr:>10.2f}  {avg_mse:>12.4f}")

    print(f"{'═' * 100}")

    feature_names = ['mean', 'std', 'grad_h', 'grad_v', 'q1', 'q2', 'q3', 'q4']
    print(f"\n  GA-optimized weights:")
    for name, w in zip(feature_names, best_weights):
        bar = '█' * int(w * 20)
        print(f"    {name:<8}  {w:>6.3f}  {bar}")

    # ============================================================
    # Save CSV
    # ============================================================
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir,
                            f"feature_weight_comparison_{timestamp}.csv")

    fieldnames = ['image', 'method', 'weight_type', 'psnr_actual', 'mse_mean',
                  'encoding_time_sec', 'feature_setup_time_sec',
                  'n_evaluations', 'ga_weights']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            if isinstance(row.get('ga_weights'), list):
                row['ga_weights'] = str(row['ga_weights'])
            writer.writerow(row)

    print(f"\n  Saved CSV: {csv_path}")

    # ============================================================
    # Save convergence data (compatible with plot_convergence.py)
    # ============================================================
    from convergence import save_convergence
    from collections import defaultdict as _dd

    conv_groups = _dd(dict)  # image → {method: curve}
    for r in all_results:
        curve = r.get('convergence_curve')
        if curve:
            conv_groups[r['image']][r['method']] = curve

    for image_name, curves in conv_groups.items():
        path = save_convergence(curves, args.output_dir, image_name, run_idx=0)
        print(f"  Saved convergence: {path}")


if __name__ == "__main__":
    main()
