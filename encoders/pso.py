"""
=============================================================================
encoders/pso.py - PSO-based FIC Encoder
=============================================================================

復現 Muruganandham & Wahida Banu 2010 的 "Adaptive Fractal Image Compression
using PSO"，使用 Particle Swarm Optimization 加速 FIC encoding。

■ 核心思想：
  對每個 range block，不窮舉所有 (domain_idx, isometry) 組合，
  而是用 PSO 在解空間中智慧搜索，大幅降低 fitness evaluation 次數。

■ 粒子編碼 (Particle Representation)：
  每個粒子是 2D 向量 [d, k]：
    d ∈ [0, N_D - 1]: domain block index (連續搜索 → 取 round 解碼)
    k ∈ [0, 7]:        isometry type      (連續搜索 → 取 round 解碼)
  s 和 o 由解析公式直接算出 (不需要搜索)，這是 Krishnamoorthy & Wu 等人
  普遍使用的設計。

■ PSO 更新公式 (Muruganandham eqn. 7-8)：
    v_{j,d}(t) = w·v_{j,d}(t-1)
                 + c1·φ1·(pbest_{j,d} - x_{j,d}(t-1))
                 + c2·φ2·(gbest_d    - x_{j,d}(t-1))
    x_{j,d}(t) = x_{j,d}(t-1) + v_{j,d}(t)
  其中 w = inertia weight, c1, c2 = acceleration coefficients,
       φ1, φ2 ~ U(0, 1)

■ Stopping criterion：
  Muruganandham 採用「gbest 連續若干代未改善」即停止 (約 max_iter 的 10%)。

■ 與 Full Search 的差異：
  - Full Search: 對每個 range block 評估 N_D × 8 = 7688 次
  - PSO:        對每個 range block 評估約 pop_size × max_iter = 40 × 30 = 1200 次
                (但實際因 early stopping 通常更少)
=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


# =============================================================================
# 粒子解碼 (連續座標 → 離散索引)
# =============================================================================

def decode_particle(particle, n_domain):
    """
    將 PSO 粒子的連續位置解碼為離散的 (domain_idx, isometry)。

    粒子 = [d, k]，d 和 k 都是連續實數，這裡取 round + clamp。
    """
    d_idx = int(np.clip(round(particle[0]), 0, n_domain - 1))
    k = int(round(particle[1])) % 8
    return d_idx, k


# =============================================================================
# 對單一 range block 跑一次 PSO
# =============================================================================

def pso_search_one_range(r_block, all_iso, n_domain,
                         pop_size=40, max_iter=30,
                         w=0.9, c1=2.0, c2=2.0,
                         v_max_ratio=0.2,
                         early_stop_patience=None,
                         ffe_budget=None,
                         rng=None):
    """
    用 PSO 為單一 range block 搜索最佳 (domain_idx, isometry)。

    Args:
        ffe_budget: int or None, 此 range block 可用的 FFE 上限。
                    None 表示無限制（僅受 max_iter 限制）。
                    達到此上限時提前終止。

    Returns:
        best: dict {'mse', 'd_idx', 'iso', 's', 'o'}
        n_evals: 此次 PSO 用掉的 fitness evaluations 數
    """
    if rng is None:
        rng = np.random.default_rng()
    if early_stop_patience is None:
        early_stop_patience = max(3, int(max_iter * 0.1))

    # --- 搜索空間範圍 ---
    # 維度 0: domain index, 範圍 [0, n_domain-1]
    # 維度 1: isometry,     範圍 [0, 7]
    x_min = np.array([0.0, 0.0])
    x_max = np.array([n_domain - 1.0, 7.0])
    v_max = v_max_ratio * (x_max - x_min)

    # --- 初始化粒子位置與速度 ---
    positions = rng.uniform(x_min, x_max, size=(pop_size, 2))
    velocities = rng.uniform(-v_max, v_max, size=(pop_size, 2))

    # --- 評估初始 fitness ---
    pbest_pos = positions.copy()
    pbest_fit = np.full(pop_size, np.inf)
    pbest_so = [(0.0, 0.0)] * pop_size  # 對應的 (s, o)

    n_evals = 0

    for j in range(pop_size):
        d_idx, k = decode_particle(positions[j], n_domain)
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
        n_evals += 1
        if abs(s) >= 1.0:
            mse = np.inf  # contractivity penalty
        pbest_fit[j] = mse
        pbest_so[j] = (s, o)

    # --- 全域最佳 ---
    g_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = pbest_fit[g_idx]
    gbest_so = pbest_so[g_idx]

    no_improve_count = 0

    # --- PSO 主迴圈 ---
    for it in range(max_iter):
        prev_gbest = gbest_fit

        # FFE budget check (在開始下一代前)
        if ffe_budget is not None and n_evals >= ffe_budget:
            break

        # 速度更新
        r1 = rng.random(size=(pop_size, 2))
        r2 = rng.random(size=(pop_size, 2))
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos[None, :] - positions))
        # 速度限幅
        velocities = np.clip(velocities, -v_max, v_max)

        # 位置更新
        positions = positions + velocities
        # 位置限幅 (邊界處理：clamp)
        positions = np.clip(positions, x_min, x_max)

        # Fitness 評估與 pbest/gbest 更新
        for j in range(pop_size):
            # 細粒度 FFE 檢查（避免超出 budget 太多）
            if ffe_budget is not None and n_evals >= ffe_budget:
                break

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

    # --- 解碼最佳解 ---
    best_d, best_iso = decode_particle(gbest_pos, n_domain)
    best_s, best_o = gbest_so

    # 防衛：若最佳解 contractivity 不合法 (理論上不該發生)
    if abs(best_s) >= 1.0:
        # 退而求其次：在 gbest 附近暴力檢查少數合法解
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


# =============================================================================
# PSO Encoder 主函式
# =============================================================================

def encode_pso(image, range_size=8, domain_size=16, domain_stride=8,
               pop_size=40, max_iter=30, w=0.9, c1=2.0, c2=2.0,
               v_max_ratio=0.2, early_stop_patience=None,
               ffe_budget_per_block=None, seed=42):
    """
    PSO-based FIC Encoder.

    Args:
        ffe_budget_per_block: int or None, 每個 range block 可用的 FFE 上限。
                              用於與其他 metaheuristic 公平比較。

    參數遵循 Muruganandham 2010：pop_size=40, max_iter=30, w=0.9
    """
    print("=" * 60)
    print("  PSO-based FIC Encoding (Muruganandham 2010)")
    print("=" * 60)
    print(f"  PSO params: pop_size={pop_size}, max_iter={max_iter}, "
          f"w={w}, c1={c1}, c2={c2}")
    if ffe_budget_per_block:
        print(f"  FFE budget: {ffe_budget_per_block} per range block")

    # 共用核心
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

    # === PSO 主迴圈：每個 range block 一次 PSO ===
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
            ffe_budget=ffe_budget_per_block,
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

    print(f"\n  ✓ PSO complete | Time: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | Avg MSE: {mean_mse:.4f}\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "images/test.png"

    stats = core.run_pipeline(
        encode_pso, path, method_name='pso',
        # 可在這裡覆寫 PSO 參數
        # pop_size=40, max_iter=30,
    )
