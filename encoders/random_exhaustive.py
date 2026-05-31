"""
=============================================================================
encoders/random_exhaustive.py - Random-K Exhaustive Search (Baseline)
=============================================================================

Control baseline for Experiment 4: randomly select K domain blocks per range
block (instead of feature-guided selection), then exhaustive search.

This directly answers: "Is it the small exhaustive search itself that helps,
or does the feature-guided pool quality matter?"

If FGDS-80 >> Random-80: feature-guided pool is valuable.
If they are similar: small exhaustive search alone is the key insight.
=============================================================================
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


def encode_random_exhaustive(image,
                             range_size=8, domain_size=16, domain_stride=8,
                             top_k=80,
                             ffe_budget_per_block=None,
                             seed=42,
                             **_ignored):
    """
    Random Domain Selection + Exhaustive Search.

    For each range block:
      1. Randomly sample top_k domain blocks (uniform, without replacement)
      2. Exhaustive search over top_k × 8 isometries

    This is the control for FGDS: same search budget, random pool instead of
    feature-guided pool.

    Returns:
        fractal_codes, encoding_time, stats, domain_positions
    """
    print("=" * 60)
    print("  Random-K Exhaustive Search (Baseline)")
    print("=" * 60)
    print(f"  top_k={top_k} → {top_k * 8} FFE per range block (exhaustive, random pool)")

    rng = np.random.RandomState(seed)

    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks: {n_range}, Domain blocks: {n_domain}")
    print(f"  Random sample: {top_k} / {n_domain} domains per range block")

    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # Pre-generate all random candidate pools (reproducible)
    k_actual = min(top_k, n_domain)
    candidate_pools = np.stack([
        rng.choice(n_domain, size=k_actual, replace=False)
        for _ in range(n_range)
    ])  # (n_range, k_actual)

    print(f"  Running exhaustive search for {n_range} range blocks...")
    fractal_codes = []
    total_evals = 0
    all_conv_logs = []
    t0 = time.time()

    for r_idx in range(n_range):
        r_block = range_blocks[r_idx]
        pool = candidate_pools[r_idx]

        best_mse = np.inf
        best_d_idx = int(pool[0])
        best_iso = 0
        best_s, best_o = 0.0, 0.0
        n_evals = 0
        conv_log = []

        for ci in range(k_actual):
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
        'feature_setup_time_sec': 0.0,
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        'convergence_curve': convergence_curve,
        'random_top_k': top_k,
        'seed': seed,
    }

    print(f"\n  Random-Exhaustive complete | "
          f"Encode: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | PSNR: {psnr:.2f} dB\n")

    return fractal_codes, encoding_time, stats, domain_positions
