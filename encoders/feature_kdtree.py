"""
=============================================================================
encoders/feature_kdtree.py - Feature-Guided Exhaustive with KD-Tree Acceleration
=============================================================================

和 feature_exhaustive.py 相同的 encoding 邏輯 (top-K × 8 exhaustive search),
但 candidate pool construction 改用 scipy.spatial.KDTree, 把 O(M×P) brute-force
降到 O(M × K × log P), 預處理從 ~180s 降到 ~3-5s。

這是我們最終提出的方法: Feature-Guided Domain Selection (FGDS)。

用法:
    python run_experiments.py --methods feature_kdtree

    # 或獨立執行
    python -m encoders.feature_kdtree images/0007.png

=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core
from encoders.feature_guided_pso import extract_features


def build_candidate_pools_kdtree(range_features, domain_features, top_k):
    """
    用 KD-Tree 加速 top-K candidate pool 建構。

    Brute-force:  O(n_range × n_domain × n_feat) ≈ 180s for 1024×1024
    KD-Tree:      O(n_domain × log(n_domain)) build + O(n_range × K × log(n_domain)) query

    在 8 維空間中 KD-Tree 仍然有效 (一般認為 d < 20 時 KD-Tree 比 brute-force 快)。

    Args:
        range_features:  (n_range, F) normalized features
        domain_features: (n_domain, F) normalized features
        top_k: int

    Returns:
        candidates: np.ndarray (n_range, top_k), dtype=int32
                    每列是 top-K domain indices, 按 feature 距離由近到遠排序
    """
    from scipy.spatial import KDTree

    n_range = range_features.shape[0]
    n_domain = domain_features.shape[0]
    k = min(top_k, n_domain)

    # 建 KD-Tree (leaf_size 調大可加速 batch query)
    tree = KDTree(domain_features, leafsize=40)

    # Batch query: 一次查全部 range blocks 的 top-K nearest neighbors
    # 回傳 (distances, indices), 都是 (n_range, k)
    _, indices = tree.query(range_features, k=k, workers=-1)

    return indices.astype(np.int32)


def encode_feature_kdtree(image,
                          range_size=8, domain_size=16, domain_stride=8,
                          top_k=40,
                          feature_normalize=True,
                          ffe_budget_per_block=None,
                          seed=42,
                          **_ignored):
    """
    Feature-Guided Domain Selection with KD-Tree acceleration.

    Pipeline:
      1. Extract 8-dim isometry-invariant features for all blocks
      2. Build candidate pools using KD-Tree (fast!)
      3. Exhaustive search within top-K × 8 isometries

    Returns:
        fractal_codes, encoding_time, stats, domain_positions
    """
    print("=" * 60)
    print("  Feature-Guided Domain Selection (KD-Tree)")
    print("=" * 60)
    print(f"  top_k={top_k} → {top_k * 8} FFE per range block (exhaustive)")

    # Extract blocks
    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks: {n_range}, Domain blocks: {n_domain}")
    print(f"  Search space reduction: {n_domain * 8} → {top_k * 8} "
          f"({n_domain * 8 / (top_k * 8):.0f}×)")

    # Precompute isometries
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

    # Stage 2: KD-Tree candidate pool construction
    print(f"  Building candidate pools via KD-Tree (top-{top_k})...", end=" ", flush=True)
    t_pool0 = time.time()
    candidate_pools = build_candidate_pools_kdtree(
        range_feats_n, domain_feats_n, top_k
    )
    t_pool = time.time() - t_pool0
    feature_setup_time = time.time() - t_feat0
    print(f"done ({t_pool:.2f}s)")
    print(f"  Total preprocessing: {feature_setup_time:.2f}s "
          f"(features: {t_feat:.2f}s + KD-Tree: {t_pool:.2f}s)")

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
        'kdtree_build_query_time_sec': round(t_pool, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        'convergence_curve': convergence_curve,
        'fgds_top_k': top_k,
    }

    print(f"\n  FGDS-KDTree complete | "
          f"Preprocess: {feature_setup_time:.2f}s | "
          f"Encode: {encoding_time:.2f}s | "
          f"Total: {feature_setup_time + encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | PSNR: {psnr:.2f} dB\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "images/0007.png"
    core.run_pipeline(
        encode_feature_kdtree, path,
        method_name='feature_kdtree',
        image_size=1024,
        range_size=4, domain_size=8, domain_stride=4,
    )
