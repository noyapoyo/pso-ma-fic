"""
=============================================================================
encoders/memetic_pso.py - Memetic Standard PSO for FIC
=============================================================================

PSO + Local Search 的 memetic 版本。
用於 ablation study：證明「LS 對 standard PSO 也有幫助」，
並對照 MPPSO 展示「PPSO 比 standard PSO 更適合作為 global search」。

與 memetic_ppso 共用相同的 LS pipeline，只是 global search 換成 standard PSO。
=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core
from encoders.pso import decode_particle
from local_search import isometry_ls, spatial_ls, LocalSearchPipeline
from local_search.spatial import build_domain_index_lookup


def memetic_pso_search_one_range(
        r_block, all_iso, n_domain,
        domain_positions, position_to_idx, domain_stride,
        pop_size=40, max_iter=30,
        w=0.9, c1=2.0, c2=2.0,
        v_max_ratio=0.2,
        early_stop_patience=None,
        ls_pipeline=None,
        ls_frequency=5,
        ls_top_percent=0.2,
        ls_at_end=True,
        ffe_budget=None,
        rng=None):
    """Memetic Standard PSO 對單一 range block 的搜索 (含 FFE budget)。"""

    if rng is None:
        rng = np.random.default_rng()
    if early_stop_patience is None:
        early_stop_patience = max(3, int(max_iter * 0.1))

    ls_context = {
        'n_domain':         n_domain,
        'domain_positions': domain_positions,
        'position_to_idx':  position_to_idx,
        'domain_stride':    domain_stride,
    }

    x_min = np.array([0.0, 0.0])
    x_max = np.array([n_domain - 1.0, 7.0])
    v_max = v_max_ratio * (x_max - x_min)

    positions = rng.uniform(x_min, x_max, size=(pop_size, 2))
    velocities = rng.uniform(-v_max, v_max, size=(pop_size, 2))

    pbest_pos = positions.copy()
    pbest_fit = np.full(pop_size, np.inf)
    pbest_so = [(0.0, 0.0)] * pop_size

    n_evals_global = 0
    n_evals_ls = 0
    ls_triggers = 0
    ls_improvements = 0

    # 初始評估
    for j in range(pop_size):
        d_idx, k = decode_particle(positions[j], n_domain)
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
        n_evals_global += 1
        if abs(s) >= 1.0:
            mse = np.inf
        pbest_fit[j] = mse
        pbest_so[j] = (s, o)

    g_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = pbest_fit[g_idx]
    gbest_so = pbest_so[g_idx]

    no_improve_count = 0
    cur_fit = pbest_fit.copy()  # 當前迭代各粒子的 fitness

    # PSO 主迴圈
    for it in range(max_iter):
        prev_gbest = gbest_fit

        # FFE budget check (global + LS 合計)
        if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
            break

        # 更新速度與位置
        r1 = rng.random(size=(pop_size, 2))
        r2 = rng.random(size=(pop_size, 2))
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos[None, :] - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions = positions + velocities
        positions = np.clip(positions, x_min, x_max)

        # 評估與 pbest/gbest 更新
        cur_so = [(0.0, 0.0)] * pop_size
        for j in range(pop_size):
            if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                break
            d_idx, k = decode_particle(positions[j], n_domain)
            s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, k)
            n_evals_global += 1
            if abs(s) >= 1.0:
                mse = np.inf
            cur_fit[j] = mse
            cur_so[j] = (s, o)

            if mse < pbest_fit[j]:
                pbest_fit[j] = mse
                pbest_pos[j] = positions[j].copy()
                pbest_so[j] = (s, o)
                if mse < gbest_fit:
                    gbest_fit = mse
                    gbest_pos = positions[j].copy()
                    gbest_so = (s, o)

        # === Local Search ===
        if ls_pipeline is not None and (it + 1) % ls_frequency == 0:
            # 若 FFE 預算將盡，跳過本次 LS
            if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                pass
            else:
                ls_triggers += 1
                n_top = max(1, int(pop_size * ls_top_percent))
                top_p_indices = np.argsort(cur_fit)[:n_top]

                for j in top_p_indices:
                    if not np.isfinite(cur_fit[j]):
                        continue
                    if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                        break

                    d_idx, iso = decode_particle(positions[j], n_domain)
                    s_cur, o_cur = cur_so[j]
                    current_sol = {
                        'd_idx': d_idx, 'iso': iso,
                        's': s_cur, 'o': o_cur, 'mse': cur_fit[j],
                    }
                    improved, n_ls, _ = ls_pipeline.apply(
                        r_block, all_iso, current_sol, ls_context
                    )
                    n_evals_ls += n_ls

                    if improved['mse'] < cur_fit[j]:
                        ls_improvements += 1
                        positions[j, 0] = float(improved['d_idx'])
                        positions[j, 1] = float(improved['iso'])
                        cur_fit[j] = improved['mse']
                    # 同步 pbest
                    if improved['mse'] < pbest_fit[j]:
                        pbest_fit[j] = improved['mse']
                        pbest_pos[j] = positions[j].copy()
                        pbest_so[j] = (improved['s'], improved['o'])
                        if improved['mse'] < gbest_fit:
                            gbest_fit = improved['mse']
                            gbest_pos = positions[j].copy()
                            gbest_so = (improved['s'], improved['o'])

        if gbest_fit < prev_gbest - 1e-10:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break

    # 結束前對 gbest 再做一次 LS
    if ls_at_end and ls_pipeline is not None:
        best_d, best_iso = decode_particle(gbest_pos, n_domain)
        current_sol = {
            'd_idx': best_d, 'iso': best_iso,
            's': gbest_so[0], 'o': gbest_so[1], 'mse': gbest_fit,
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

    best_d, best_iso = decode_particle(gbest_pos, n_domain)
    best_s, best_o = gbest_so

    # 防衛
    if abs(best_s) >= 1.0:
        for d_idx in range(max(0, best_d - 2), min(n_domain, best_d + 3)):
            for iso in range(8):
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
                n_evals_global += 1
                if abs(s) < 1.0 and mse < gbest_fit:
                    gbest_fit, best_d, best_iso, best_s, best_o = \
                        mse, d_idx, iso, s, o

    return {
        'mse': float(gbest_fit), 'd_idx': int(best_d), 'iso': int(best_iso),
        's': float(best_s), 'o': float(best_o),
    }, {
        'n_evals_global': n_evals_global, 'n_evals_ls': n_evals_ls,
        'ls_triggers': ls_triggers, 'ls_improvements': ls_improvements,
    }


def encode_memetic_pso(image, range_size=8, domain_size=16, domain_stride=8,
                       pop_size=40, max_iter=30,
                       w=0.9, c1=2.0, c2=2.0, v_max_ratio=0.2,
                       early_stop_patience=None,
                       ls_strategies=('isometry', 'spatial'),
                       ls_frequency=5,
                       ls_top_percent=0.2,
                       ls_at_end=True,
                       ffe_budget_per_block=None,
                       seed=42):
    """Memetic Standard PSO based FIC Encoder."""

    print("=" * 60)
    print("  Memetic Standard PSO (M-PSO) FIC Encoding")
    print("=" * 60)
    print(f"  PSO params:    pop={pop_size}, max_iter={max_iter}, "
          f"w={w}, c1={c1}, c2={c2}")
    print(f"  LS strategies: {list(ls_strategies)}")
    if ffe_budget_per_block:
        print(f"  FFE budget:    {ffe_budget_per_block} per range block")

    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks:  {n_range}, Domain blocks: {n_domain}\n")
    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # LS pipeline
    ls_funcs = {'isometry': isometry_ls, 'spatial': spatial_ls}
    ls_pipeline = None
    if ls_strategies:
        ls_pipeline = LocalSearchPipeline()
        for name in ls_strategies:
            ls_pipeline.add(name, ls_funcs[name])
    position_to_idx = build_domain_index_lookup(
        domain_positions, image.shape, domain_size, domain_stride
    )

    print(f"  Running M-PSO for {n_range} range blocks...")
    fractal_codes = []
    total_g, total_l, total_t, total_i = 0, 0, 0, 0
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for r_idx in range(n_range):
        best, ss = memetic_pso_search_one_range(
            range_blocks[r_idx], all_iso, n_domain,
            domain_positions, position_to_idx, domain_stride,
            pop_size=pop_size, max_iter=max_iter, w=w, c1=c1, c2=c2,
            v_max_ratio=v_max_ratio,
            early_stop_patience=early_stop_patience,
            ls_pipeline=ls_pipeline,
            ls_frequency=ls_frequency, ls_top_percent=ls_top_percent,
            ls_at_end=ls_at_end,
            ffe_budget=ffe_budget_per_block,
            rng=rng,
        )
        total_g += ss['n_evals_global']
        total_l += ss['n_evals_ls']
        total_t += ss['ls_triggers']
        total_i += ss['ls_improvements']

        fractal_codes.append({
            'range_pos': range_positions[r_idx],
            'domain_idx': best['d_idx'],
            'domain_pos': domain_positions[best['d_idx']],
            'isometry': best['iso'],
            'contrast': best['s'], 'brightness': best['o'],
            'mse': best['mse'],
        })

        if (r_idx + 1) % 128 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    encoding_time = time.time() - t0
    total_evals = total_g + total_l
    all_mse = [c['mse'] for c in fractal_codes]
    mean_mse = float(np.mean(all_mse))
    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else float('inf')

    stats = {
        'n_range': n_range, 'n_domain': n_domain,
        'n_evaluations': total_evals,
        'n_evals_global': total_g, 'n_evals_ls': total_l,
        'ls_triggers': total_t, 'ls_improvements': total_i,
        'encoding_time_sec': round(encoding_time, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        'mpso_pop_size': pop_size, 'mpso_max_iter': max_iter,
        'mpso_ls': list(ls_strategies),
    }

    print(f"\n  ✓ M-PSO complete | Time: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | PSNR: {psnr:.2f} dB\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "images/test.png"
    core.run_pipeline(encode_memetic_pso, path, method_name='memetic_pso')
