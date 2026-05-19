"""
=============================================================================
encoders/ppso.py - Pyramid PSO based FIC Encoder
=============================================================================

實作 Pyramid Particle Swarm Optimization (PPSO) 用於 FIC encoding。
PPSO 的核心 idea 來自 Li et al. (2022)：

  將粒子依 fitness 排序後分配到金字塔的不同層級，
  同層粒子兩兩配對比較產生 winner/loser，
  Loser 向 winner 學習，Winner 向上層 + 頂層全域最佳粒子學習。

這種「分層競爭合作」的機制提供了天然的 exploration-exploitation 平衡：
  - 頂層 (好的粒子)：精細 exploitation
  - 底層 (差的粒子)：大幅 exploration

■ 金字塔結構 (4 層, swarm_size=30):
              [2]   ← Top tier (elite, 全域最佳)
             [4]    ← Tier 2
           [8]      ← Tier 3
         [16]       ← Bottom tier (探索)

■ 與標準 PSO 的關鍵差異：
  - Standard PSO: 所有粒子都向同一個 gbest 學習
  - PPSO:        每個粒子有「分層 exemplars」，多元學習對象

■ Implementation 參考：
  Li, Y., et al. (2022). "A pyramid particle swarm optimization with
  novel strategies of competition and cooperation." Expert Systems
  with Applications.
=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


# =============================================================================
# 工具函式：粒子解碼
# =============================================================================

def decode_particle(particle, n_domain):
    """連續位置 → (domain_idx, isometry)。"""
    d_idx = int(np.clip(round(particle[0]), 0, n_domain - 1))
    k = int(round(particle[1])) % 8
    return d_idx, k


# =============================================================================
# 金字塔結構工具
# =============================================================================

def build_pyramid_structure(swarm_size):
    """
    決定金字塔每層的粒子數。
    預設 4 層: [bottom, tier3, tier2, top]，例如 swarm=30 → [16, 8, 4, 2]

    Returns:
        layer_sizes: list of int, 由底到頂 (例如 [16, 8, 4, 2])
        layer_ranges: list of (start, end), 每層在 sorted index 中的範圍
    """
    if swarm_size == 30:
        # 標準配置 (Li et al. 推薦)
        layer_sizes = [16, 8, 4, 2]  # 由底到頂
    elif swarm_size == 20:
        layer_sizes = [10, 6, 3, 1]
    elif swarm_size == 40:
        layer_sizes = [20, 12, 6, 2]
    else:
        # 一般化：用 2:1 遞減比例
        layer_sizes = []
        remaining = swarm_size
        size = max(2, swarm_size // 2)
        while remaining > 0 and len(layer_sizes) < 4:
            actual = min(size, remaining)
            layer_sizes.append(actual)
            remaining -= actual
            size = max(1, size // 2)
        if remaining > 0:
            layer_sizes[-1] += remaining

    # 由底到頂的 (start, end) 索引範圍
    # 排序後 index 0 = 最佳，所以「頂層」對應索引 [0, layer_top_size)
    # 為了直觀，我們用「由頂到底」的順序去切 sorted indices
    layer_sizes_top_down = list(reversed(layer_sizes))  # [top, tier2, tier3, bottom]
    layer_ranges_top_down = []
    cur = 0
    for sz in layer_sizes_top_down:
        layer_ranges_top_down.append((cur, cur + sz))
        cur += sz

    return layer_sizes_top_down, layer_ranges_top_down


def assign_particles_to_layers(fitness_array, layer_ranges):
    """
    依 fitness 排序，將粒子索引分配到各層。
    Returns: list of arrays, layer_particles[i] = i-th 層的粒子原始索引
    layer_particles[0] = 頂層 (最佳粒子)
    """
    sorted_idx = np.argsort(fitness_array)  # 升序：MSE 越小越好
    layer_particles = []
    for start, end in layer_ranges:
        layer_particles.append(sorted_idx[start:end].copy())
    return layer_particles


# =============================================================================
# PPSO 更新公式
# =============================================================================

def update_winner(particle_idx, winner_target_idx, top_tier_idx,
                  positions, velocities, w, c1, c2, c3, rng,
                  v_max):
    """
    Winner 的更新規則（往上層 + 頂層學習）：

      v(t+1) = w·v(t)
               + c1·r1·(x_upper - x)        ← 向上層 winner 學習
               + c2·r2·(x_top   - x)        ← 向頂層全域最佳學習
      x(t+1) = x(t) + v(t+1)

    Args:
        particle_idx:        當前 winner 的索引
        winner_target_idx:   上層對應 winner 的索引 (學習對象)
        top_tier_idx:        頂層粒子之一的索引 (全域 elite)
        positions, velocities: swarm 的位置與速度陣列
    """
    pos = positions[particle_idx]
    vel = velocities[particle_idx]
    target_upper = positions[winner_target_idx]
    target_top = positions[top_tier_idx]

    r1 = rng.random(size=2)
    r2 = rng.random(size=2)

    new_vel = (w * vel
               + c1 * r1 * (target_upper - pos)
               + c2 * r2 * (target_top - pos))
    new_vel = np.clip(new_vel, -v_max, v_max)

    return new_vel


def update_loser(particle_idx, winner_idx,
                 positions, velocities, w, c1, rng, v_max):
    """
    Loser 的更新規則（向同層 winner 學習）：

      v(t+1) = w·v(t) + c1·r1·(x_winner - x)
      x(t+1) = x(t) + v(t+1)
    """
    pos = positions[particle_idx]
    vel = velocities[particle_idx]
    target = positions[winner_idx]

    r1 = rng.random(size=2)
    new_vel = w * vel + c1 * r1 * (target - pos)
    new_vel = np.clip(new_vel, -v_max, v_max)

    return new_vel


def random_pair_within_layer(layer_indices, rng):
    """
    將同層粒子隨機兩兩配對。
    若粒子數為奇數，最後一個落單者單獨返回。
    Returns:
        pairs: list of (idx_a, idx_b)
        unpaired: list of int (落單的粒子索引)
    """
    indices = list(layer_indices)
    rng.shuffle(indices)
    pairs = []
    unpaired = []
    i = 0
    while i + 1 < len(indices):
        pairs.append((indices[i], indices[i + 1]))
        i += 2
    if i < len(indices):
        unpaired.append(indices[i])
    return pairs, unpaired


# =============================================================================
# 對單一 range block 跑一次 PPSO
# =============================================================================

def ppso_search_one_range(r_block, all_iso, n_domain,
                          pop_size=30, max_iter=30,
                          w=0.7, c1=1.5, c2=1.5, c3=1.0,
                          v_max_ratio=0.2,
                          early_stop_patience=None,
                          ffe_budget=None,
                          rng=None):
    """
    用 PPSO 為單一 range block 搜索最佳 (domain_idx, isometry)。

    Args:
        r_block:    range block (8x8)
        all_iso:    預計算的 isometry 查表
        n_domain:   domain pool 大小
        pop_size:   粒子數 (預設 30，配合金字塔 [2,4,8,16])
        max_iter:   最大迭代數
        w:          inertia weight
        c1, c2:     同層/頂層學習係數
        ffe_budget: int or None, FFE 上限。達到後提前終止。

    Returns:
        best:    dict {'mse', 'd_idx', 'iso', 's', 'o'}
        n_evals: fitness evaluation 次數
    """
    if rng is None:
        rng = np.random.default_rng()
    if early_stop_patience is None:
        early_stop_patience = max(3, int(max_iter * 0.1))

    # --- 搜索空間 ---
    x_min = np.array([0.0, 0.0])
    x_max = np.array([n_domain - 1.0, 7.0])
    v_max = v_max_ratio * (x_max - x_min)

    # --- 金字塔結構 ---
    layer_sizes, layer_ranges = build_pyramid_structure(pop_size)
    n_layers = len(layer_sizes)  # 通常 4

    # --- 初始化粒子 ---
    positions = rng.uniform(x_min, x_max, size=(pop_size, 2))
    velocities = rng.uniform(-v_max, v_max, size=(pop_size, 2))
    fitness = np.full(pop_size, np.inf)
    so_cache = [(0.0, 0.0)] * pop_size  # 對應的 (s, o)

    n_evals = 0

    # --- 評估初始 fitness ---
    for j in range(pop_size):
        d_idx, k = decode_particle(positions[j], n_domain)
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
        n_evals += 1
        if abs(s) >= 1.0:
            mse = np.inf
        fitness[j] = mse
        so_cache[j] = (s, o)

    # --- 全域最佳 (gbest) ---
    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]
    gbest_so = so_cache[gbest_idx]

    no_improve_count = 0

    # --- PPSO 主迴圈 ---
    for it in range(max_iter):
        prev_gbest = gbest_fit

        # FFE budget check
        if ffe_budget is not None and n_evals >= ffe_budget:
            break

        # 依 fitness 重新排序、分層
        layer_particles = assign_particles_to_layers(fitness, layer_ranges)
        # layer_particles[0] = 頂層 (最佳粒子們)
        # layer_particles[-1] = 底層 (最差粒子們)

        top_indices = layer_particles[0]  # 頂層粒子索引

        # --- 對每一層做 winner/loser 競爭 + 學習 ---
        # 收集本輪要更新的速度（保留 elitism：頂層粒子不更新位置）
        new_velocities = velocities.copy()
        winner_in_each_layer = []  # 各層的 winners (用於下一層 loser 學習)

        for layer_idx in range(n_layers):
            layer = layer_particles[layer_idx]

            # 同層兩兩配對
            pairs, unpaired = random_pair_within_layer(layer, rng)

            layer_winners = []

            for idx_a, idx_b in pairs:
                if fitness[idx_a] <= fitness[idx_b]:
                    winner, loser = idx_a, idx_b
                else:
                    winner, loser = idx_b, idx_a

                layer_winners.append(winner)

                # === 更新 Loser ===
                # Loser 向同層 winner 學習
                new_velocities[loser] = update_loser(
                    loser, winner, positions, velocities,
                    w, c1, rng, v_max
                )

                # === 更新 Winner ===
                # 頂層 winner 不更新（elitism），其他層 winner 向上層學習
                if layer_idx == 0:
                    # 頂層 elitism：保留位置和速度
                    pass
                else:
                    # 非頂層 winner：向上層 winner + 頂層 elite 學習
                    upper_layer_winners = winner_in_each_layer[layer_idx - 1] \
                        if winner_in_each_layer else top_indices.tolist()
                    if upper_layer_winners:
                        upper_target = rng.choice(upper_layer_winners)
                    else:
                        upper_target = top_indices[0]
                    top_target = rng.choice(top_indices)

                    new_velocities[winner] = update_winner(
                        winner, upper_target, top_target,
                        positions, velocities,
                        w, c1, c2, c3, rng, v_max
                    )

            # 落單的粒子處理：當 winner 處理（向上層學習，或頂層保留）
            for solo in unpaired:
                if layer_idx == 0:
                    pass  # elitism
                else:
                    upper_layer_winners = winner_in_each_layer[layer_idx - 1] \
                        if winner_in_each_layer else top_indices.tolist()
                    upper_target = rng.choice(upper_layer_winners) \
                        if upper_layer_winners else top_indices[0]
                    top_target = rng.choice(top_indices)
                    new_velocities[solo] = update_winner(
                        solo, upper_target, top_target,
                        positions, velocities,
                        w, c1, c2, c3, rng, v_max
                    )
                layer_winners.append(solo)

            winner_in_each_layer.append(layer_winners)

        # === 套用速度更新位置（頂層除外，elitism）===
        velocities = new_velocities
        for j in range(pop_size):
            if j in top_indices:
                continue  # elitism: 頂層不動
            positions[j] = positions[j] + velocities[j]
        positions = np.clip(positions, x_min, x_max)

        # === 重新評估 fitness（除了頂層）===
        for j in range(pop_size):
            if j in top_indices:
                continue
            if ffe_budget is not None and n_evals >= ffe_budget:
                break
            d_idx, k = decode_particle(positions[j], n_domain)
            s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
            n_evals += 1
            if abs(s) >= 1.0:
                mse = np.inf
            fitness[j] = mse
            so_cache[j] = (s, o)

        # === 更新全域最佳 ===
        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < gbest_fit:
            gbest_fit = fitness[cur_best_idx]
            gbest_pos = positions[cur_best_idx].copy()
            gbest_so = so_cache[cur_best_idx]

        # === Early stopping ===
        if gbest_fit < prev_gbest - 1e-10:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break

    # --- 解碼最佳解 ---
    best_d, best_iso = decode_particle(gbest_pos, n_domain)
    best_s, best_o = gbest_so

    # 防衛性處理：若最佳解 contractivity 不合法
    if abs(best_s) >= 1.0:
        for d_idx in range(max(0, best_d - 2), min(n_domain, best_d + 3)):
            for iso in range(8):
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                n_evals += 1
                if abs(s) < 1.0 and mse < gbest_fit:
                    gbest_fit, best_d, best_iso, best_s, best_o = \
                        mse, d_idx, iso, s, o

    return {
        'mse': float(gbest_fit),
        'd_idx': int(best_d),
        'iso': int(best_iso),
        's': float(best_s),
        'o': float(best_o),
    }, n_evals


# =============================================================================
# PPSO Encoder 主函式
# =============================================================================

def encode_ppso(image, range_size=8, domain_size=16, domain_stride=8,
                pop_size=30, max_iter=30,
                w=0.7, c1=1.5, c2=1.5, c3=1.0,
                v_max_ratio=0.2, early_stop_patience=None,
                ffe_budget_per_block=None, seed=42):
    """
    Pyramid PSO based FIC Encoder。

    Args:
        ffe_budget_per_block: int or None, 每個 range block 的 FFE 上限。
    """
    print("=" * 60)
    print("  PPSO-based FIC Encoding (Pyramid PSO)")
    print("=" * 60)
    layer_sizes, _ = build_pyramid_structure(pop_size)
    print(f"  Pyramid structure (top→bottom): {layer_sizes}")
    print(f"  Params: pop_size={pop_size}, max_iter={max_iter}, "
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
    print()

    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    print(f"  Running PPSO for {n_range} range blocks...")
    fractal_codes = []
    total_evals = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for r_idx in range(n_range):
        best, n_evals = ppso_search_one_range(
            range_blocks[r_idx], all_iso, n_domain,
            pop_size=pop_size, max_iter=max_iter,
            w=w, c1=c1, c2=c2, c3=c3,
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
        # PPSO-specific
        'ppso_pop_size': pop_size,
        'ppso_max_iter': max_iter,
        'ppso_pyramid': layer_sizes,
        'ppso_w': w,
        'ppso_c1': c1,
        'ppso_c2': c2,
    }

    print(f"\n  ✓ PPSO complete | Time: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | Avg MSE: {mean_mse:.4f}\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "images/test.png"

    stats = core.run_pipeline(
        encode_ppso, path, method_name='ppso'
    )
