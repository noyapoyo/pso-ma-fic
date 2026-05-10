"""
PSO-based FIC Encoder

Implementation of Muruganandham & Wahida Banu 2010 "Adaptive Fractal Image Compression
using PSO.

Reference:
* A. Muruganandham and R.S.D. Wahida Banu, "Adaptive Fractal Image Compression using PSO," 
Procedia Computer Science, vol. 2, pp. 338-344, 2010. [ICEBT 2010]. 
DOI: [10.1016/j.procs.2010.11.044](https://doi.org/10.1016/j.procs.2010.11.044)
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


def decode_particle(particle, n_domain):
    """
    Decode PSO Particle continuous position to discreate (domain_idx, isometry).

    Particle = [d, k]，d and k are continuous real number, round + clamp.
    """
    d_idx = int(np.clip(round(particle[0]), 0, n_domain - 1))
    k = int(round(particle[1])) % 8
    return d_idx, k


def pso_search_one_range(r_block, all_iso, n_domain,
                         pop_size=40, max_iter=30,
                         w=0.9, c1=2.0, c2=2.0,
                         v_max_ratio=0.2,
                         early_stop_patience=None,
                         rng=None):
    """
    Utilize PSO to search single range block optimal (domain_idx, isometry)。

    Args:
        r_block:    range block (8x8)
        all_iso:    Precompute isometry lookup table (n_domain, 8, 8, 8)
        n_domain:   domain pool size
        pop_size:   Particle size (Muruganandham use 40)
        max_iter:   Maximum iteration (Muruganandham use 30)
        w:          inertia weight (Muruganandham use 0.9)
        c1, c2:     acceleration coefficients (classic: 2.0)
        v_max_ratio: velocity maximum = v_max_ratio × search range 
        early_stop_patience: gbest continuous unimproved variable -> stop 
                             None represent max_iter * 10% (Muruganandham suggest)

    Returns:
        best: dict {'mse', 'd_idx', 'iso', 's', 'o'}
        n_evals: In this iteration, the fitness evaluations PSO used.
    """
    if rng is None:
        rng = np.random.default_rng()
    if early_stop_patience is None:
        early_stop_patience = max(3, int(max_iter * 0.1))

    # Search space and range
    # dim 0: domain index, range [0, n_domain-1]
    # dim 1: isometry,     range [0, 7]
    x_min = np.array([0.0, 0.0])
    x_max = np.array([n_domain - 1.0, 7.0])
    v_max = v_max_ratio * (x_max - x_min)

    # Initialize Particle position and velocity
    positions = rng.uniform(x_min, x_max, size=(pop_size, 2))
    velocities = rng.uniform(-v_max, v_max, size=(pop_size, 2))

    # Evaluate initial fitness
    pbest_pos = positions.copy()
    pbest_fit = np.full(pop_size, np.inf)
    pbest_so = [(0.0, 0.0)] * pop_size  # (s, o)

    n_evals = 0

    for j in range(pop_size):
        d_idx, k = decode_particle(positions[j], n_domain)
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
        n_evals += 1
        if abs(s) >= 1.0:
            mse = np.inf  # contractivity penalty
        pbest_fit[j] = mse
        pbest_so[j] = (s, o)

    # global optima
    g_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = pbest_fit[g_idx]
    gbest_so = pbest_so[g_idx]

    no_improve_count = 0

    # PSO loop
    for it in range(max_iter):
        prev_gbest = gbest_fit

        # velocity updated 
        r1 = rng.random(size=(pop_size, 2))
        r2 = rng.random(size=(pop_size, 2))
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos[None, :] - positions))

        velocities = np.clip(velocities, -v_max, v_max)

        # position updated
        positions = positions + velocities
        # 位置限幅 (邊界處理：clamp)
        positions = np.clip(positions, x_min, x_max)

        # Fitness evaluation and pbest/gbest update
        for j in range(pop_size):
            d_idx, k = decode_particle(positions[j], n_domain)
            s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
            n_evals += 1
            if abs(s) >= 1.0:
                mse = np.inf

            if mse < pbest_fit[j]:
                pbest_fit[j] = mse
                pbest_pos[j] = positions[j].copy()
                pbest_so[j] = (s, o)
                if mse < gbest_fit:
                    gbest_fit = mse
                    gbest_pos = positions[j].copy()
                    gbest_so = (s, o)

        # Early stopping
        if gbest_fit < prev_gbest - 1e-10:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break

    # Optimal solution
    best_d, best_iso = decode_particle(gbest_pos, n_domain)
    best_s, best_o = gbest_so

    # if the optimal solution contractivity illegal
    if abs(best_s) >= 1.0:
        # brute force to find gbest neighbor for legal solution
        for d_idx in range(max(0, best_d - 2), min(n_domain, best_d + 3)):
            for iso in range(8):
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                n_evals += 1
                if abs(s) < 1.0 and mse < gbest_fit:
                    gbest_fit, best_d, best_iso, best_s, best_o = mse, d_idx, iso, s, o

    return {
        'mse': float(gbest_fit),
        'd_idx': int(best_d),
        'iso': int(best_iso),
        's': float(best_s),
        'o': float(best_o),
    }, n_evals


def encode_pso(image, range_size=8, domain_size=16, domain_stride=8,
               pop_size=40, max_iter=30, w=0.9, c1=2.0, c2=2.0,
               v_max_ratio=0.2, early_stop_patience=None, seed=42):
    """
    PSO-based FIC Encoder.

    For each range block execute one time PSO to find the optimal (d_idx, isometry).

    Muruganandham 2010：pop_size=40, max_iter=30, w=0.9
    """
    print("=" * 60)
    print("  PSO-based FIC Encoding (Muruganandham 2010)")
    print("=" * 60)
    print(f"  PSO params: pop_size={pop_size}, max_iter={max_iter}, "
          f"w={w}, c1={c1}, c2={c2}")

    # Extract blocks 
    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks: {n_range}, Domain blocks: {n_domain}")
    print(f"  Max evals per range: ~{pop_size * (max_iter + 1)} "
          f"(vs Full Search: {n_domain * 8:,})")
    print()

    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # PSO main loop: run PSO per range block 
    print(f"  Running PSO for {n_range} range blocks...")
    fractal_codes = []
    total_evals = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for r_idx in range(n_range):
        best, n_evals = pso_search_one_range(
            range_blocks[r_idx], all_iso, n_domain,
            pop_size=pop_size, max_iter=max_iter,
            w=w, c1=c1, c2=c2,
            v_max_ratio=v_max_ratio,
            early_stop_patience=early_stop_patience,
            rng=rng,
        )
        total_evals += n_evals

        fractal_codes.append({
            'range_pos': range_positions[r_idx],
            'domain_idx': best['d_idx'],
            'domain_pos': domain_positions[best['d_idx']],
            'isometry': best['iso'],
            'contrast': best['s'],
            'brightness': best['o'],
            'mse': best['mse'],
        })

        if (r_idx + 1) % 128 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            avg_evals = total_evals / (r_idx + 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  "
                  f"avg_evals/block={avg_evals:.0f}")

    encoding_time = time.time() - t0

    all_mse = [c['mse'] for c in fractal_codes]
    mean_mse = float(np.mean(all_mse))
    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else float('inf')

    stats = {
        'n_range': n_range,
        'n_domain': n_domain,
        'n_evaluations': total_evals,
        'encoding_time_sec': round(encoding_time, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        # PSO-specific
        'pso_pop_size': pop_size,
        'pso_max_iter': max_iter,
        'pso_w': w,
        'pso_c1': c1,
        'pso_c2': c2,
    }

    print(f"\n  PSO complete | Time: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | Avg MSE: {mean_mse:.4f}\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "images/test.png"

    stats = core.run_pipeline(
        encode_pso, path, method_name='pso',
        # pop_size=40, max_iter=30,
    )
