"""
=============================================================================
encoders/feature_exhaustive.py - Feature-Only Exhaustive Search (Ablation)
=============================================================================

Ablation baseline: 用 FG-PSO 的 feature extraction + candidate pool,
但不用 PSO, 直接對 top-K domains × 8 isometries 做 exhaustive search。

如果 K=40, 總共 40×8 = 320 FFE < B=500 budget。
這個 ablation 回答的問題是:
  「PSO + LS 在 candidate pool 內是否有貢獻, 還是 feature pre-filtering 就夠了?」

介面完全相容 run_experiments.py。
=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core
from encoders.feature_guided_pso import extract_features, build_candidate_pools


def encode_feature_exhaustive(image,
                              range_size=8, domain_size=16, domain_stride=8,
                              top_k=40,
                              feature_normalize=True,
                              ffe_budget_per_block=None,
                              seed=42,
                              # 以下參數是為了相容 run_experiments 的 kwargs 傳遞,
                              # 本方法不使用, 但要接住避免 TypeError
                              **_ignored):
    """
    Feature-Only Exhaustive Search for FIC encoding.

    對每個 range block:
      1. 用 feature distance 找到 top-K candidate domains
      2. 對 K × 8 = K*8 個 (domain, isometry) 組合做 exhaustive search
      3. 取 MSE 最小的那個 (contractivity |s| < 1)

    No PSO, no local search.

    Returns:
        fractal_codes, encoding_time, stats, domain_positions
    """
    print("=" * 60)
    print("  Feature-Only Exhaustive Search (Ablation)")
    print("=" * 60)
    print(f"  top_k={top_k} → {top_k * 8} FFE per range block (exhaustive)")

    # 抽取 range / domain blocks
    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks: {n_range}, Domain blocks: {n_domain}")

    # 預計算 isometries
    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # Stage 1: Feature extraction
    print("  Extracting features...", end=" ", flush=True)
    t_feat0 = time.time()
    range_feats = extract_features(range_blocks)
    domain_feats = extract_features(domain_blocks)

    if feature_normalize:
        mu = domain_feats.mean(axis=0)
        sigma = domain_feats.std(axis=0) + 1e-8
        domain_feats_n = (domain_feats - mu) / sigma
        range_feats_n = (range_feats - mu) / sigma
    else:
        domain_feats_n = domain_feats
        range_feats_n = range_feats
    t_feat = time.time() - t_feat0
    print(f"done ({t_feat:.2f}s)")

    # Stage 2: Candidate pool (brute-force pairwise)
    print(f"  Building candidate pools (top-{top_k}, pairwise)...", end=" ", flush=True)
    t_pool0 = time.time()
    candidate_pools = build_candidate_pools(range_feats_n, domain_feats_n, top_k)
    t_pool = time.time() - t_pool0
    feature_setup_time = time.time() - t_feat0
    print(f"done ({t_pool:.2f}s)")
    print(f"  Total preprocessing: {feature_setup_time:.2f}s "
          f"(features: {t_feat:.2f}s + pairwise: {t_pool:.2f}s)")

    # Stage 3: Exhaustive search within candidate pool
    print(f"  Running exhaustive search for {n_range} range blocks...")
    fractal_codes = []
    total_evals = 0
    all_conv_logs = []
    t0 = time.time()

    for r_idx in range(n_range):
        r_block = range_blocks[r_idx]
        pool = candidate_pools[r_idx]
        k_cand = len(pool)

        best_mse = np.inf
        best_d_idx = int(pool[0])
        best_iso = 0
        best_s, best_o = 0.0, 0.0
        n_evals = 0
        conv_log = []

        for ci in range(k_cand):
            d_idx = int(pool[ci])
            for iso in range(8):
                # 如果有 FFE budget 且已超過, 停止
                if ffe_budget_per_block is not None and n_evals >= ffe_budget_per_block:
                    break

                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                n_evals += 1

                if abs(s) >= 1.0:
                    continue
                if mse < best_mse:
                    best_mse = mse
                    best_d_idx = d_idx
                    best_iso = iso
                    best_s, best_o = float(s), float(o)
                    conv_log.append((n_evals, float(best_mse)))

            if ffe_budget_per_block is not None and n_evals >= ffe_budget_per_block:
                break

        if not conv_log:
            conv_log.append((n_evals, float(best_mse)))
        all_conv_logs.append(conv_log)
        total_evals += n_evals

        fractal_codes.append({
            'range_pos': range_positions[r_idx],
            'domain_idx': best_d_idx,
            'domain_pos': domain_positions[best_d_idx],
            'isometry': best_iso,
            'contrast': best_s,
            'brightness': best_o,
            'mse': float(best_mse),
        })

        if (r_idx + 1) % 256 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    encoding_time = time.time() - t0

    # Convergence curve
    from convergence import aggregate_convergence
    budget = ffe_budget_per_block or (top_k * 8)
    convergence_curve = aggregate_convergence(all_conv_logs, budget)

    all_mse = [c['mse'] for c in fractal_codes]
    mean_mse = float(np.mean(all_mse))
    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else float('inf')

    stats = {
        'n_range': n_range, 'n_domain': n_domain,
        'n_evaluations': total_evals,
        'n_evals_global': total_evals, 'n_evals_ls': 0,
        'ls_triggers': 0, 'ls_improvements': 0,
        'encoding_time_sec': round(encoding_time, 3),
        'feature_setup_time_sec': round(feature_setup_time, 3),
        'feature_extract_time_sec': round(t_feat, 3),
        'pairwise_pool_time_sec': round(t_pool, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        'convergence_curve': convergence_curve,
        'ablation_top_k': top_k,
    }

    print(f"\n  Feature-Exhaustive complete | Time: {encoding_time:.2f}s "
          f"(feature setup: {feature_setup_time:.2f}s) | "
          f"Evals: {total_evals:,} | PSNR: {psnr:.2f} dB\n")

    return fractal_codes, encoding_time, stats, domain_positions
