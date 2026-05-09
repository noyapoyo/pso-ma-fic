import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


def encode_full_search(image, range_size=8, domain_size=16, domain_stride=8):
    """
    Full Search FIC Encoding.

    Returns:
        fractal_codes: list of dicts
        encoding_time: float (sec)
        stats: dict
        domain_positions: list of (row, col)
    """
    print("=" * 60)
    print("  Full Search FIC Encoding")
    print("=" * 60)

    # Extract blocks
    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )

    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks: {n_range}, Domain blocks: {n_domain}")
    print(f"  Total fitness evals: {n_range * n_domain * 8:,}")
    print()

    # isometry
    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # Full Search main
    print(f"  Encoding {n_range} range blocks...")
    fractal_codes = []
    n_evals = 0
    t0 = time.time()

    for r_idx in range(n_range):
        r_block = range_blocks[r_idx]
        best = {'mse': float('inf'), 'd_idx': 0, 'iso': 0, 's': 0.0, 'o': 0.0}

        for d_idx in range(n_domain):
            for iso in range(8):
                # fitness evaluation
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                n_evals += 1
                if abs(s) >= 1.0:  # contractivity
                    continue
                if mse < best['mse']:
                    best = {'mse': mse, 'd_idx': d_idx, 'iso': iso, 's': s, 'o': o}

        fractal_codes.append({
            'range_pos': range_positions[r_idx],
            'domain_idx': best['d_idx'],
            'domain_pos': domain_positions[best['d_idx']],
            'isometry': best['iso'],
            'contrast': float(best['s']),
            'brightness': float(best['o']),
            'mse': float(best['mse']),
        })

        if (r_idx + 1) % 128 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    encoding_time = time.time() - t0

    all_mse = [c['mse'] for c in fractal_codes]
    mean_mse = float(np.mean(all_mse))
    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else float('inf')

    stats = {
        'n_range': n_range,
        'n_domain': n_domain,
        'n_evaluations': n_evals,
        'encoding_time_sec': round(encoding_time, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
    }

    print(f"\n  Full Search complete | Time: {encoding_time:.2f}s | "
          f"Evals: {n_evals:,} | Avg MSE: {mean_mse:.4f}\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "images/test.png"

    stats = core.run_pipeline(
        encode_full_search, path, method_name='full_search'
    )
