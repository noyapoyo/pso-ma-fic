"""
=============================================================================
encoders/memetic_ppso.py - Memetic Pyramid PSO (MPPSO) for FIC ⭐
=============================================================================

本論文的 proposed method：Memetic Pyramid PSO (MPPSO)。

■ 演算法概觀：
  將 Pyramid PSO 的全域搜索能力，與 FIC-specific Local Search 結合，
  形成一個 memetic framework：
    Global Search (PPSO)  +  Local Refinement (Isometry LS + Spatial LS)

■ 完整流程：

  Input: range block R, isometry lookup table, domain pool
  Output: best fractal code (d_idx, iso, s, o, mse)

  ┌─────────────────────────────────────────────────────────────┐
  │ Step 1: PPSO 初始化                                          │
  │   - 30 個粒子隨機分布在搜索空間                              │
  │   - 計算每個粒子的 fitness (MSE)                             │
  │   - 建立金字塔分層 [16, 8, 4, 2]                             │
  └─────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 2: PPSO 主迴圈 (每代)                                    │
  │   2a. 依 fitness 重新分層                                    │
  │   2b. 同層粒子兩兩配對 → winner / loser                      │
  │   2c. Loser 向同層 winner 學習                                │
  │   2d. Winner 向上層 winner + 頂層 elite 學習                  │
  │   2e. 頂層 elitism (保留)                                    │
  │   2f. 重新評估 fitness, 更新 gbest                           │
  └─────────────────────────────────────────────────────────────┘
                       │
                       │ 每隔 ls_frequency 代觸發
                       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 3: Local Search Refinement                              │
  │   3a. 選取 fitness 排名前 ls_top_percent 的粒子              │
  │   3b. 對每個被選中的粒子：                                    │
  │       ┌──────────────────────────────────────────┐         │
  │       │ Sub-step 3b-1: Isometry LS               │         │
  │       │   - 固定 d_idx, 遍歷 7 種其他 isometry    │         │
  │       │   - 找到更好的 isometry 就更新            │         │
  │       └──────────────────────────────────────────┘         │
  │       ┌──────────────────────────────────────────┐         │
  │       │ Sub-step 3b-2: Spatial LS                │         │
  │       │   - 檢查 8 個空間鄰居                     │         │
  │       │   - 找到更好的 domain 就更新              │         │
  │       └──────────────────────────────────────────┘         │
  │   3c. 將精煉後的解寫回粒子位置（更新 fitness）                │
  │   3d. 重新評估 gbest                                          │
  └─────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Step 4: 終止條件檢查                                          │
  │   - 達到 max_iter，或                                        │
  │   - gbest 連續 patience 代未改善                              │
  │   若都不滿足，回到 Step 2                                     │
  └─────────────────────────────────────────────────────────────┘

■ LS 設計細節：
  - LS 不是每代都做（太慢），預設每 5 代做一次
  - LS 只對 top 20% 粒子做（成本控制）
  - Isometry LS 在前，Spatial LS 在後（先便宜後昂貴）

■ 與其他方法的差異：
  - vs Full Search: 用智慧搜索取代窮舉
  - vs PSO:         金字塔多元學習 + 局部精煉
  - vs PPSO:        多了局部精煉這層
  - vs Memetic-PSO: 全域搜索用 PPSO 而非 standard PSO
=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core
from encoders.ppso import (
    decode_particle, build_pyramid_structure,
    assign_particles_to_layers,
    update_winner, update_loser, random_pair_within_layer,
)
from local_search import isometry_ls, spatial_ls, LocalSearchPipeline
from local_search.spatial import build_domain_index_lookup


# =============================================================================
# 對單一 range block 跑一次 Memetic PPSO
# =============================================================================

def memetic_ppso_search_one_range(
        r_block, all_iso, n_domain,
        domain_positions, position_to_idx, domain_stride,
        # PPSO 參數
        pop_size=30, max_iter=30,
        w=0.7, c1=1.5, c2=1.5, c3=1.0,
        v_max_ratio=0.2,
        early_stop_patience=None,
        # LS 參數
        ls_pipeline=None,
        ls_frequency=5,
        ls_top_percent=0.2,
        ls_at_end=True,
        # FFE budget
        ffe_budget=None,
        rng=None):
    """
    Memetic Pyramid PSO 對單一 range block 的搜索。

    結合 PPSO 全域搜索 + Local Search 局部精煉。

    Args:
        r_block:            range block
        all_iso:            isometry 查表
        n_domain:           domain pool 大小
        domain_positions:   domain 位置列表 (LS 用)
        position_to_idx:    位置→索引反查表 (Spatial LS 用)
        domain_stride:      domain 步長 (Spatial LS 用)
        ls_pipeline:        LocalSearchPipeline 實例 (None 表示不做 LS)
        ls_frequency:       每幾代做一次 LS
        ls_top_percent:     LS 對前幾 % 粒子做
        ls_at_end:          結束前對 gbest 再做一次 LS（保險）

    Returns:
        best: dict {'mse', 'd_idx', 'iso', 's', 'o'}
        stats: dict {'n_evals_global', 'n_evals_ls', 'ls_triggers',
                     'ls_improvements'}
    """
    if rng is None:
        rng = np.random.default_rng()
    if early_stop_patience is None:
        early_stop_patience = max(3, int(max_iter * 0.1))

    # LS context：傳給 LS 算子的共用參數
    ls_context = {
        'n_domain':         n_domain,
        'domain_positions': domain_positions,
        'position_to_idx':  position_to_idx,
        'domain_stride':    domain_stride,
    }

    # --- 搜索空間 ---
    x_min = np.array([0.0, 0.0])
    x_max = np.array([n_domain - 1.0, 7.0])
    v_max = v_max_ratio * (x_max - x_min)

    # --- 金字塔結構 ---
    layer_sizes, layer_ranges = build_pyramid_structure(pop_size)
    n_layers = len(layer_sizes)

    # --- 初始化 ---
    positions = rng.uniform(x_min, x_max, size=(pop_size, 2))
    velocities = rng.uniform(-v_max, v_max, size=(pop_size, 2))
    fitness = np.full(pop_size, np.inf)
    so_cache = [(0.0, 0.0)] * pop_size

    n_evals_global = 0
    n_evals_ls = 0
    ls_triggers = 0
    ls_improvements = 0

    # 評估初始 fitness
    for j in range(pop_size):
        d_idx, k = decode_particle(positions[j], n_domain)
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
        n_evals_global += 1
        if abs(s) >= 1.0:
            mse = np.inf
        fitness[j] = mse
        so_cache[j] = (s, o)

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]
    gbest_so = so_cache[gbest_idx]

    no_improve_count = 0

    # === PPSO 主迴圈 ===
    for it in range(max_iter):
        prev_gbest = gbest_fit

        # FFE budget check
        if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
            break

        layer_particles = assign_particles_to_layers(fitness, layer_ranges)
        top_indices = layer_particles[0]

        new_velocities = velocities.copy()
        winner_in_each_layer = []

        for layer_idx in range(n_layers):
            layer = layer_particles[layer_idx]
            pairs, unpaired = random_pair_within_layer(layer, rng)
            layer_winners = []

            for idx_a, idx_b in pairs:
                if fitness[idx_a] <= fitness[idx_b]:
                    winner, loser = idx_a, idx_b
                else:
                    winner, loser = idx_b, idx_a
                layer_winners.append(winner)

                new_velocities[loser] = update_loser(
                    loser, winner, positions, velocities,
                    w, c1, rng, v_max
                )

                if layer_idx == 0:
                    pass  # elitism
                else:
                    upper_winners = winner_in_each_layer[layer_idx - 1] \
                        if winner_in_each_layer else top_indices.tolist()
                    upper_target = rng.choice(upper_winners) \
                        if upper_winners else top_indices[0]
                    top_target = rng.choice(top_indices)
                    new_velocities[winner] = update_winner(
                        winner, upper_target, top_target,
                        positions, velocities,
                        w, c1, c2, c3, rng, v_max
                    )

            for solo in unpaired:
                if layer_idx == 0:
                    pass
                else:
                    upper_winners = winner_in_each_layer[layer_idx - 1] \
                        if winner_in_each_layer else top_indices.tolist()
                    upper_target = rng.choice(upper_winners) \
                        if upper_winners else top_indices[0]
                    top_target = rng.choice(top_indices)
                    new_velocities[solo] = update_winner(
                        solo, upper_target, top_target,
                        positions, velocities,
                        w, c1, c2, c3, rng, v_max
                    )
                layer_winners.append(solo)

            winner_in_each_layer.append(layer_winners)

        # 套用速度更新位置（除頂層）
        velocities = new_velocities
        for j in range(pop_size):
            if j in top_indices:
                continue
            positions[j] = positions[j] + velocities[j]
        positions = np.clip(positions, x_min, x_max)

        # 重新評估 fitness
        for j in range(pop_size):
            if j in top_indices:
                continue
            if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                break
            d_idx, k = decode_particle(positions[j], n_domain)
            s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
            n_evals_global += 1
            if abs(s) >= 1.0:
                mse = np.inf
            fitness[j] = mse
            so_cache[j] = (s, o)

        # === Local Search Refinement (每 ls_frequency 代觸發) ===
        if ls_pipeline is not None and (it + 1) % ls_frequency == 0:
            # 若 FFE 預算將盡，跳過 LS
            if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                pass
            else:
                ls_triggers += 1

                # 選取 top p% 粒子
                n_top = max(1, int(pop_size * ls_top_percent))
                top_p_indices = np.argsort(fitness)[:n_top]

                for j in top_p_indices:
                    if not np.isfinite(fitness[j]):
                        continue
                    if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                        break

                    # 把粒子 j 的當前位置包成 LS 的 input 格式
                    d_idx, iso = decode_particle(positions[j], n_domain)
                    s_cur, o_cur = so_cache[j]
                    current_sol = {
                        'd_idx': d_idx,
                        'iso':   iso,
                        's':     s_cur,
                        'o':     o_cur,
                        'mse':   fitness[j],
                    }

                    # 套用 LS pipeline
                    improved, n_ls, _ = ls_pipeline.apply(
                        r_block, all_iso, current_sol, ls_context
                    )
                    n_evals_ls += n_ls

                    # 若 LS 找到更好解 → 寫回粒子
                    if improved['mse'] < fitness[j]:
                        ls_improvements += 1
                        positions[j, 0] = float(improved['d_idx'])
                        positions[j, 1] = float(improved['iso'])
                        fitness[j] = improved['mse']
                        so_cache[j] = (improved['s'], improved['o'])

        # 更新全域最佳
        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < gbest_fit:
            gbest_fit = fitness[cur_best_idx]
            gbest_pos = positions[cur_best_idx].copy()
            gbest_so = so_cache[cur_best_idx]

        # Early stopping
        if gbest_fit < prev_gbest - 1e-10:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break

    # --- 結束前對 gbest 再做一次 LS（保險）---
    if ls_at_end and ls_pipeline is not None:
        best_d, best_iso = decode_particle(gbest_pos, n_domain)
        current_sol = {
            'd_idx': best_d,
            'iso':   best_iso,
            's':     gbest_so[0],
            'o':     gbest_so[1],
            'mse':   gbest_fit,
        }
        improved, n_ls, _ = ls_pipeline.apply(
            r_block, all_iso, current_sol, ls_context
        )
        n_evals_ls += n_ls
        if improved['mse'] < gbest_fit:
            ls_improvements += 1
            gbest_fit = improved['mse']
            gbest_pos = np.array([float(improved['d_idx']),
                                  float(improved['iso'])])
            gbest_so = (improved['s'], improved['o'])

    # 解碼最終解
    best_d, best_iso = decode_particle(gbest_pos, n_domain)
    best_s, best_o = gbest_so

    # 防衛：contractivity
    if abs(best_s) >= 1.0:
        for d_idx in range(max(0, best_d - 2), min(n_domain, best_d + 3)):
            for iso in range(8):
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                n_evals_global += 1
                if abs(s) < 1.0 and mse < gbest_fit:
                    gbest_fit, best_d, best_iso, best_s, best_o = \
                        mse, d_idx, iso, s, o

    return {
        'mse':   float(gbest_fit),
        'd_idx': int(best_d),
        'iso':   int(best_iso),
        's':     float(best_s),
        'o':     float(best_o),
    }, {
        'n_evals_global':  n_evals_global,
        'n_evals_ls':      n_evals_ls,
        'ls_triggers':     ls_triggers,
        'ls_improvements': ls_improvements,
    }


# =============================================================================
# Memetic PPSO Encoder 主函式
# =============================================================================

def encode_memetic_ppso(image, range_size=8, domain_size=16, domain_stride=8,
                        # PPSO 參數
                        pop_size=30, max_iter=30,
                        w=0.7, c1=1.5, c2=1.5, c3=1.0,
                        v_max_ratio=0.2, early_stop_patience=None,
                        # LS 配置
                        ls_strategies=('isometry', 'spatial'),
                        ls_frequency=5,
                        ls_top_percent=0.2,
                        ls_at_end=True,
                        # FFE budget
                        ffe_budget_per_block=None,
                        seed=42):
    """
    Memetic Pyramid PSO based FIC Encoder. ⭐

    Args:
        ls_strategies:        LS 算子順序，可用 'isometry', 'spatial'
        ls_frequency:         每幾代觸發 LS
        ls_top_percent:       LS 對 top % 粒子做
        ls_at_end:            結束前對 gbest 做最後 LS
        ffe_budget_per_block: 每 range block 的 FFE 上限 (公平比較用)
    """
    print("=" * 60)
    print("  Memetic Pyramid PSO (MPPSO) FIC Encoding")
    print("=" * 60)
    layer_sizes, _ = build_pyramid_structure(pop_size)
    print(f"  Pyramid:       {layer_sizes}  (top → bottom)")
    print(f"  PPSO params:   pop={pop_size}, max_iter={max_iter}, "
          f"w={w}, c1={c1}, c2={c2}")
    print(f"  LS strategies: {list(ls_strategies)}")
    print(f"  LS schedule:   every {ls_frequency} iter, "
          f"top {int(ls_top_percent*100)}%, at_end={ls_at_end}")

    # --- 提取 blocks ---
    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks:  {n_range}")
    print(f"  Domain blocks: {n_domain}")
    print()

    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # --- 準備 LS Pipeline ---
    ls_funcs = {
        'isometry': isometry_ls,
        'spatial':  spatial_ls,
    }
    ls_pipeline = None
    if ls_strategies:
        ls_pipeline = LocalSearchPipeline()
        for name in ls_strategies:
            if name not in ls_funcs:
                raise ValueError(f"Unknown LS strategy: {name}")
            ls_pipeline.add(name, ls_funcs[name])
    position_to_idx = build_domain_index_lookup(
        domain_positions, image.shape, domain_size, domain_stride
    )

    # --- 主迴圈 ---
    print(f"  Running MPPSO for {n_range} range blocks...")
    fractal_codes = []
    total_evals_global = 0
    total_evals_ls = 0
    total_ls_triggers = 0
    total_ls_improvements = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for r_idx in range(n_range):
        best, search_stats = memetic_ppso_search_one_range(
            range_blocks[r_idx], all_iso, n_domain,
            domain_positions, position_to_idx, domain_stride,
            pop_size=pop_size, max_iter=max_iter,
            w=w, c1=c1, c2=c2, c3=c3,
            v_max_ratio=v_max_ratio,
            early_stop_patience=early_stop_patience,
            ls_pipeline=ls_pipeline,
            ls_frequency=ls_frequency,
            ls_top_percent=ls_top_percent,
            ls_at_end=ls_at_end,
            ffe_budget=ffe_budget_per_block,
            rng=rng,
        )
        total_evals_global    += search_stats['n_evals_global']
        total_evals_ls        += search_stats['n_evals_ls']
        total_ls_triggers     += search_stats['ls_triggers']
        total_ls_improvements += search_stats['ls_improvements']

        fractal_codes.append({
            'range_pos':  range_positions[r_idx],
            'domain_idx': best['d_idx'],
            'domain_pos': domain_positions[best['d_idx']],
            'isometry':   best['iso'],
            'contrast':   best['s'],
            'brightness': best['o'],
            'mse':        best['mse'],
        })

        if (r_idx + 1) % 128 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            avg_evals = (total_evals_global + total_evals_ls) / (r_idx + 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  "
                  f"avg_evals={avg_evals:.0f}")

    encoding_time = time.time() - t0
    total_evals = total_evals_global + total_evals_ls

    all_mse = [c['mse'] for c in fractal_codes]
    mean_mse = float(np.mean(all_mse))
    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else float('inf')

    ls_pct = 100 * total_evals_ls / total_evals if total_evals > 0 else 0
    imp_rate = 100 * total_ls_improvements / max(total_ls_triggers, 1)

    stats = {
        'n_range':         n_range,
        'n_domain':        n_domain,
        'n_evaluations':   total_evals,
        'n_evals_global':  total_evals_global,
        'n_evals_ls':      total_evals_ls,
        'ls_overhead_pct': round(ls_pct, 2),
        'ls_triggers':     total_ls_triggers,
        'ls_improvements': total_ls_improvements,
        'ls_improvement_rate_pct': round(imp_rate, 2),
        'encoding_time_sec': round(encoding_time, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max':  round(float(np.max(all_mse)), 4),
        'psnr_db':  round(psnr, 2),
        # MPPSO 參數
        'mppso_pop_size':    pop_size,
        'mppso_max_iter':    max_iter,
        'mppso_pyramid':     layer_sizes,
        'mppso_ls':          list(ls_strategies),
        'mppso_ls_freq':     ls_frequency,
        'mppso_ls_top_pct':  ls_top_percent,
    }

    print(f"\n  ✓ MPPSO complete | Time: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} (LS overhead {ls_pct:.1f}%) | "
          f"PSNR: {psnr:.2f} dB")
    print(f"  LS stats: {total_ls_triggers} triggers, "
          f"{total_ls_improvements} improvements "
          f"({imp_rate:.1f}% success rate)\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "images/test.png"

    stats = core.run_pipeline(
        encode_memetic_ppso, path, method_name='memetic_ppso'
    )
