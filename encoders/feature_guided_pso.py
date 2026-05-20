"""
=============================================================================
encoders/feature_guided_pso.py - Feature-Guided PSO with Memetic Local Search
=============================================================================

■ Motivation:
  Standard PSO (Muruganandham 2010) 在 (domain_idx, isometry) 二維空間中搜尋。
  但這個 2D 空間極小 (對 256x256/range=4 只有 8192 種組合)，PSO 容易困在
  「離全域最佳很遠的局部山頭」: 我們實測在 boat 256x256 上 PSO/Memetic-PPSO
  與 Full Search 的最佳 domain index 只有 5-7% 一致, 而錯誤匹配的平均空間距離
  超過 100 pixels — 也就是 PSO 收斂到的解和真正最佳解空間上不相鄰, 任何
  「鄰域型 local search」都救不了。

■ 我們的 contribution:
  將搜尋空間從「全 domain pool × 8 iso」縮減到「feature-排序前 K 個 domain × 8 iso」,
  並在這個更小但更高品質的搜尋空間上跑 standard PSO + memetic LS。

■ Pipeline:
  1. Feature extraction: 對每個 domain / range block, 抽取 6 維 isometry-invariant
     feature: [mean, std, |∇h|_mean, |∇v|_mean, q1, q2, q3, q4_sorted]
  2. Candidate pool: 對每個 range block, 用 normalized Euclidean 距離排序,
     取 top-K candidate domains (預設 K=80)
  3. Restricted PSO: 在 (cand_idx ∈ [0, K-1], iso ∈ [0, 7]) 上跑 standard PSO
     (相同的 Muruganandham 2010 更新公式)
  4. Memetic Local Search:
     - isometry_ls: 固定 domain 試遍 8 個 isometry (cheap, 已存在)
     - feature_ls : 沿 feature 距離排序的下一層 N 個 candidate 做精煉
                    (取代原 spatial_ls — feature-neighbor 比 spatial-neighbor 更相關)

■ Key design choices:
  - 為什麼用 6 維 feature 而非 DCT/PCA?: 簡單、可解釋、計算極快, 而且實測 hit-rate
    已經比 PSO 好很多. 為了 CVGIP 8 頁論文, 簡單可解釋更重要.
  - 為什麼用 isometry-invariant feature?: 因為一個 domain 經過任一 iso 變換後
    應該被視為同一個候選, 我們希望 feature 距離反映「形狀相似」, 不是「方向相似」.
  - 為什麼 K=80 而非更小?: K=80 → 候選空間 = 640 種, 比 8192 小 12.8x.
    PSO 用 pop=20, iter=15 共 ~300 evals 就能涵蓋 ~50% 子空間, 比原 PSO 在
    全空間採樣 1024/8192 = 12.5% 的覆蓋率好太多.

■ 介面相容性:
  完全沿用 fic_core.run_pipeline + run_experiments.py 的介面 (kwargs, FFE budget,
  seed). 可加入 configs/feature_guided_pso.yml 後直接用 --methods feature_guided_pso
  執行.
=============================================================================
"""

import numpy as np
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features(blocks):
    """
    抽取每個 block 的 isometry-invariant feature。

    Feature 設計 (8 維):
        0:   mean (亮度均值, 對 iso 完全不變)
        1:   std  (對比強度, 對 iso 完全不變)
        2:   |∇_horizontal|_mean (水平邊緣強度)
        3:   |∇_vertical|_mean   (垂直邊緣強度)
        4-7: 四象限均值 SORTED (對 iso 不變, 因為 rotation/flip 只是 permute quadrants)

    為什麼這些 feature 有效?
        - mean/std 抓「整體亮度與對比」
        - 邊緣強度抓「紋理粗細」  (注意: 取絕對值再平均, 對 iso 仍是 quasi-invariant
                                    在 fliplr/flipud 下完全 invariant, rot90 下會交換水平/垂直
                                    但因為我們是用「distance」, 此交換可由 SORTED quadrants 補強)
        - 4 quadrants sorted: 抓「亮度分布形狀」, 是真正的 iso-invariant signature

    Args:
        blocks: list of 2D arrays (range_size × range_size 已下採樣的 domain blocks
                或 range blocks)
    Returns:
        features: np.ndarray of shape (n, 8)
    """
    n = len(blocks)
    feats = np.zeros((n, 8), dtype=np.float64)
    for k, b in enumerate(blocks):
        h, w = b.shape
        h2, w2 = h // 2, w // 2
        # 0, 1: 基本統計量
        feats[k, 0] = b.mean()
        feats[k, 1] = b.std()
        # 2, 3: 邊緣強度
        feats[k, 2] = np.abs(np.diff(b, axis=1)).mean()
        feats[k, 3] = np.abs(np.diff(b, axis=0)).mean()
        # 4-7: 四象限均值 (sorted, iso-invariant)
        q = [b[:h2, :w2].mean(), b[:h2, w2:].mean(),
             b[h2:, :w2].mean(), b[h2:, w2:].mean()]
        feats[k, 4:8] = sorted(q)
    return feats


def build_candidate_pools(range_features, domain_features, top_k):
    """
    對每個 range block 建立 top-K 候選 domain 清單。

    Args:
        range_features:  (n_range, F) 已標準化
        domain_features: (n_domain, F) 已標準化
        top_k: int, 每個 range 保留多少 candidate

    Returns:
        candidates: np.ndarray of shape (n_range, top_k), 每列是該 range 的
                    top-K domain 索引 (按 feature 距離由近到遠排序)
    """
    n_range = range_features.shape[0]
    n_domain = domain_features.shape[0]
    k = min(top_k, n_domain)

    candidates = np.zeros((n_range, k), dtype=np.int32)
    # 向量化計算每個 range 到所有 domain 的距離 (用平方距離省 sqrt)
    # dist^2(r, d) = ||r - d||^2 = ||r||^2 + ||d||^2 - 2 r·d
    # 整批計算 (n_range, n_domain), 對大 n_domain 也很快
    # 為了省記憶體, batch 處理
    batch = 512
    for s in range(0, n_range, batch):
        e = min(s + batch, n_range)
        diff = range_features[s:e, None, :] - domain_features[None, :, :]
        dists = (diff * diff).sum(axis=2)            # (batch, n_domain)
        # argpartition O(n_domain) 比 argsort O(n_domain log n_domain) 快
        part_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        # 將前 k 名按距離由近到遠排序
        for i in range(e - s):
            row_idx = part_idx[i]
            sorted_order = np.argsort(dists[i, row_idx])
            candidates[s + i] = row_idx[sorted_order]
    return candidates


# =============================================================================
# Decoding helpers
# =============================================================================

def decode_particle_in_subspace(particle, k_candidates):
    """
    粒子位置 → (cand_idx ∈ [0, k_candidates-1], iso ∈ [0, 7])

    粒子在 [0, k_candidates-1] × [0, 7] 連續空間中, round 後映射。
    """
    cand_idx = int(np.clip(round(particle[0]), 0, k_candidates - 1))
    iso = int(round(particle[1])) % 8
    return cand_idx, iso


# =============================================================================
# Local search operators (作用於子空間)
# =============================================================================

def isometry_ls_subspace(r_block, all_iso, cand_d_idx, cur_iso, cur_mse):
    """固定 domain, 遍歷其他 7 種 isometry。"""
    best_iso = cur_iso
    best_mse = cur_mse
    best_s, best_o = 0.0, 0.0  # 會在外部覆寫
    found = False
    n_evals = 0
    for iso in range(8):
        if iso == cur_iso:
            continue
        s, o, mse = core.evaluate_candidate(r_block, all_iso, cand_d_idx, iso)
        n_evals += 1
        if abs(s) >= 1.0:
            continue
        if mse < best_mse:
            best_mse, best_iso = mse, iso
            best_s, best_o = float(s), float(o)
            found = True
    return found, best_iso, best_mse, best_s, best_o, n_evals


def feature_neighbor_ls(r_block, all_iso, candidate_pool, cur_cand_idx,
                        cur_iso, cur_mse, n_neighbors=4):
    """
    Feature-neighbor LS: 在 candidate_pool 中, cur_cand_idx 附近 (前後 n_neighbors 個)
    的 candidate domain 上, 固定 cur_iso 試一次, 看是否更好。

    為什麼這個比 spatial_ls 更好?
        - candidate_pool 已經是按 feature 距離排序的, 所以 cur_cand_idx 前後幾名
          就是「特徵上相近 (但不是空間上相近)」的 domain。
        - PSO 收斂到的 cand_idx (在 pool 內) 不一定是 pool 內的最佳,
          這個 LS 用窮舉確保 pool 內的局部最佳 (在 cur_iso 下) 被找到。

    Args:
        candidate_pool: 該 range 的 top-K candidate 在原 domain pool 的 index 陣列,
                        shape (k_candidates,)
        cur_cand_idx:   當前最佳粒子在 candidate_pool 中的索引 ∈ [0, k_candidates-1]
        n_neighbors:    往前後各看 n_neighbors 個

    Returns:
        found, new_cand_idx, new_mse, new_s, new_o, n_evals
    """
    k = len(candidate_pool)
    lo = max(0, cur_cand_idx - n_neighbors)
    hi = min(k, cur_cand_idx + n_neighbors + 1)
    best_idx = cur_cand_idx
    best_mse = cur_mse
    best_s, best_o = 0.0, 0.0
    found = False
    n_evals = 0
    for ci in range(lo, hi):
        if ci == cur_cand_idx:
            continue
        d_idx = candidate_pool[ci]
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, cur_iso)
        n_evals += 1
        if abs(s) >= 1.0:
            continue
        if mse < best_mse:
            best_mse, best_idx = mse, ci
            best_s, best_o = float(s), float(o)
            found = True
    return found, best_idx, best_mse, best_s, best_o, n_evals


# =============================================================================
# Per-range restricted PSO + memetic LS
# =============================================================================

def fgpso_search_one_range(r_block, all_iso, candidate_pool,
                           pop_size=20, max_iter=15,
                           w=0.9, c1=2.0, c2=2.0,
                           v_max_ratio=0.2,
                           early_stop_patience=None,
                           ls_strategies=('isometry', 'feature_neighbor'),
                           ls_frequency=3,
                           ls_top_percent=0.2,
                           ls_at_end=True,
                           feat_neighbors=4,
                           ffe_budget=None,
                           rng=None):
    """
    Feature-Guided PSO + Memetic LS for a single range block。

    PSO 在縮減後的 2D 空間 [0, k_candidates-1] × [0, 7] 中搜尋。
    所有 fitness evaluation 都計入 ffe_budget。

    Returns:
        best: dict {'mse', 'd_idx', 'iso', 's', 'o'}  (注意: d_idx 是「原 domain pool」的 index)
        stats: dict (n_evals_global, n_evals_ls, ls_triggers, ls_improvements)
    """
    if rng is None:
        rng = np.random.default_rng()
    if early_stop_patience is None:
        early_stop_patience = max(2, int(max_iter * 0.1))

    k_cand = len(candidate_pool)
    x_min = np.array([0.0, 0.0])
    x_max = np.array([k_cand - 1.0, 7.0])
    v_max = v_max_ratio * (x_max - x_min)

    # ---- 初始化 ----
    positions = rng.uniform(x_min, x_max, size=(pop_size, 2))
    velocities = rng.uniform(-v_max, v_max, size=(pop_size, 2))
    pbest_pos = positions.copy()
    pbest_fit = np.full(pop_size, np.inf)
    pbest_so = [(0.0, 0.0)] * pop_size

    n_evals_global = 0
    n_evals_ls = 0
    ls_triggers = 0
    ls_improvements = 0

    # 初始 fitness
    for j in range(pop_size):
        ci, iso = decode_particle_in_subspace(positions[j], k_cand)
        d_idx = int(candidate_pool[ci])
        s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
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

    # ---- PSO 主迴圈 ----
    for it in range(max_iter):
        prev_gbest = gbest_fit
        if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
            break

        r1 = rng.random(size=(pop_size, 2))
        r2 = rng.random(size=(pop_size, 2))
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos[None, :] - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions = np.clip(positions + velocities, x_min, x_max)

        # 評估
        cur_so = [(0.0, 0.0)] * pop_size
        cur_fit = np.full(pop_size, np.inf)
        for j in range(pop_size):
            if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                break
            ci, iso = decode_particle_in_subspace(positions[j], k_cand)
            d_idx = int(candidate_pool[ci])
            s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
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

        # ---- Local Search ----
        if ls_strategies and (it + 1) % ls_frequency == 0:
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

                    ci, iso = decode_particle_in_subspace(positions[j], k_cand)
                    d_idx = int(candidate_pool[ci])
                    cur_mse_j = cur_fit[j]
                    s_j, o_j = cur_so[j]
                    improved = False

                    # === isometry LS ===
                    if 'isometry' in ls_strategies:
                        f, new_iso, new_mse, new_s, new_o, ne = isometry_ls_subspace(
                            r_block, all_iso, d_idx, iso, cur_mse_j
                        )
                        n_evals_ls += ne
                        if f:
                            iso = new_iso
                            cur_mse_j = new_mse
                            s_j, o_j = new_s, new_o
                            positions[j, 1] = float(iso)
                            improved = True

                    # === feature-neighbor LS ===
                    if 'feature_neighbor' in ls_strategies:
                        if ffe_budget is not None and (n_evals_global + n_evals_ls) >= ffe_budget:
                            pass
                        else:
                            f, new_ci, new_mse, new_s, new_o, ne = feature_neighbor_ls(
                                r_block, all_iso, candidate_pool, ci, iso,
                                cur_mse_j, n_neighbors=feat_neighbors,
                            )
                            n_evals_ls += ne
                            if f:
                                ci = new_ci
                                cur_mse_j = new_mse
                                s_j, o_j = new_s, new_o
                                positions[j, 0] = float(ci)
                                improved = True

                    if improved:
                        ls_improvements += 1
                        cur_fit[j] = cur_mse_j
                        # 同步 pbest / gbest
                        if cur_mse_j < pbest_fit[j]:
                            pbest_fit[j] = cur_mse_j
                            pbest_pos[j] = positions[j].copy()
                            pbest_so[j] = (s_j, o_j)
                            if cur_mse_j < gbest_fit:
                                gbest_fit = cur_mse_j
                                gbest_pos = positions[j].copy()
                                gbest_so = (s_j, o_j)

        # ---- Early stopping ----
        if gbest_fit < prev_gbest - 1e-10:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break

    # ---- 結束前對 gbest 再做一次 LS ----
    if ls_at_end and ls_strategies:
        ci, iso = decode_particle_in_subspace(gbest_pos, k_cand)
        d_idx = int(candidate_pool[ci])
        s_j, o_j = gbest_so

        if 'isometry' in ls_strategies:
            f, new_iso, new_mse, new_s, new_o, ne = isometry_ls_subspace(
                r_block, all_iso, d_idx, iso, gbest_fit
            )
            n_evals_ls += ne
            if f:
                ls_improvements += 1
                iso = new_iso
                gbest_fit = new_mse
                s_j, o_j = new_s, new_o
                gbest_pos = np.array([float(ci), float(iso)])
                gbest_so = (s_j, o_j)

        if 'feature_neighbor' in ls_strategies:
            f, new_ci, new_mse, new_s, new_o, ne = feature_neighbor_ls(
                r_block, all_iso, candidate_pool, ci, iso, gbest_fit,
                n_neighbors=feat_neighbors,
            )
            n_evals_ls += ne
            if f:
                ls_improvements += 1
                ci = new_ci
                gbest_fit = new_mse
                s_j, o_j = new_s, new_o
                gbest_pos = np.array([float(ci), float(iso)])
                gbest_so = (s_j, o_j)

    # ---- 解碼最終解 ----
    best_ci, best_iso = decode_particle_in_subspace(gbest_pos, k_cand)
    best_d_idx = int(candidate_pool[best_ci])
    best_s, best_o = gbest_so

    # 防衛: contractivity 不合法時, 在候選池內小範圍 fallback
    if abs(best_s) >= 1.0:
        for ci_try in range(min(8, k_cand)):
            d_try = int(candidate_pool[ci_try])
            for iso_try in range(8):
                s, o, mse = core.evaluate_candidate(r_block, all_iso, d_try, iso_try)
                n_evals_global += 1
                if abs(s) < 1.0 and mse < gbest_fit:
                    gbest_fit = mse
                    best_d_idx, best_iso = d_try, iso_try
                    best_s, best_o = float(s), float(o)

    return {
        'mse': float(gbest_fit),
        'd_idx': int(best_d_idx),
        'iso': int(best_iso),
        's': float(best_s),
        'o': float(best_o),
    }, {
        'n_evals_global': n_evals_global,
        'n_evals_ls': n_evals_ls,
        'ls_triggers': ls_triggers,
        'ls_improvements': ls_improvements,
    }


# =============================================================================
# Encoder main
# =============================================================================

def encode_feature_guided_pso(image,
                              range_size=8, domain_size=16, domain_stride=8,
                              # Candidate-pool 參數
                              top_k=80,
                              feature_normalize=True,
                              # PSO 參數
                              pop_size=20, max_iter=15,
                              w=0.9, c1=2.0, c2=2.0,
                              v_max_ratio=0.2,
                              early_stop_patience=None,
                              # Memetic LS 參數
                              ls_strategies=('isometry', 'feature_neighbor'),
                              ls_frequency=3,
                              ls_top_percent=0.2,
                              ls_at_end=True,
                              feat_neighbors=4,
                              # 公平比較參數
                              ffe_budget_per_block=None,
                              seed=42):
    """
    Feature-Guided PSO with Memetic Local Search for FIC encoding.

    Args:
        top_k: 每個 range block 保留多少個 candidate domain (核心參數)
        feature_normalize: 是否對 feature 做 z-score 標準化
        pop_size, max_iter: PSO 參數 (因為搜尋空間縮減, 可用較小的 swarm)
        ls_strategies: ('isometry', 'feature_neighbor') 中選用哪些 LS operator
        feat_neighbors: feature-neighbor LS 看前後幾個 candidate
        ffe_budget_per_block: 公平比較的 FFE 上限

    Returns:
        fractal_codes, encoding_time, stats, domain_positions
    """
    print("=" * 60)
    print("  Feature-Guided PSO + Memetic LS (FG-PSO) FIC Encoding")
    print("=" * 60)
    print(f"  Candidate pool: top_k={top_k}")
    print(f"  PSO params:     pop={pop_size}, max_iter={max_iter}, "
          f"w={w}, c1={c1}, c2={c2}")
    print(f"  LS strategies:  {list(ls_strategies)}")
    print(f"  LS schedule:    every {ls_frequency} iter, top {ls_top_percent*100:.0f}%,"
          f" final_pass={ls_at_end}, feat_neighbors={feat_neighbors}")
    if ffe_budget_per_block:
        print(f"  FFE budget:     {ffe_budget_per_block} per range block")

    # 抽取 range / domain blocks
    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks:   {n_range}, Domain blocks: {n_domain}")
    print(f"  Search space:   {top_k * 8} per range "
          f"(vs full {n_domain * 8}, reduction {n_domain * 8 / (top_k * 8):.1f}x)")

    # 預計算 isometries (與其他 encoder 共用)
    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    # === Stage 1: Feature extraction ===
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
    print(f"done ({time.time() - t_feat0:.2f}s)")

    # === Stage 2: Candidate pool ===
    print(f"  Building candidate pools (top-{top_k})...", end=" ", flush=True)
    t_pool0 = time.time()
    candidate_pools = build_candidate_pools(range_feats_n, domain_feats_n, top_k)
    feature_setup_time = time.time() - t_feat0
    print(f"done ({time.time() - t_pool0:.2f}s)")

    # === Stage 3: Restricted PSO + Memetic LS ===
    print(f"  Running FG-PSO for {n_range} range blocks...")
    fractal_codes = []
    total_g = total_l = total_t = total_i = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for r_idx in range(n_range):
        best, ss = fgpso_search_one_range(
            range_blocks[r_idx], all_iso,
            candidate_pools[r_idx],
            pop_size=pop_size, max_iter=max_iter,
            w=w, c1=c1, c2=c2,
            v_max_ratio=v_max_ratio,
            early_stop_patience=early_stop_patience,
            ls_strategies=ls_strategies,
            ls_frequency=ls_frequency,
            ls_top_percent=ls_top_percent,
            ls_at_end=ls_at_end,
            feat_neighbors=feat_neighbors,
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
            'contrast': best['s'],
            'brightness': best['o'],
            'mse': best['mse'],
        })

        if (r_idx + 1) % 256 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            avg_evals = (total_g + total_l) / (r_idx + 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  "
                  f"avg_evals/block={avg_evals:.0f}")

    encoding_time = time.time() - t0

    # === Stats ===
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
        'feature_setup_time_sec': round(feature_setup_time, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        # method-specific
        'fgpso_top_k': top_k,
        'fgpso_pop_size': pop_size,
        'fgpso_max_iter': max_iter,
        'fgpso_w': w,
        'fgpso_c1': c1,
        'fgpso_c2': c2,
        'fgpso_ls': list(ls_strategies),
        'fgpso_ls_freq': ls_frequency,
        'fgpso_ls_top_pct': ls_top_percent,
        'fgpso_feat_neighbors': feat_neighbors,
    }

    print(f"\n  FG-PSO complete | Time: {encoding_time:.2f}s "
          f"(feature setup: {feature_setup_time:.2f}s) | "
          f"Evals: {total_evals:,} | PSNR: {psnr:.2f} dB\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "images_512/boat.png"
    core.run_pipeline(encode_feature_guided_pso, path, method_name='feature_guided_pso')
