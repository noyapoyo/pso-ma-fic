"""
=============================================================================
local_search/spatial.py - Spatial Neighborhood Local Search (C1)
=============================================================================

■ 核心 idea：
  在當前最佳 domain block 的「空間鄰域」中搜索更好的匹配。

■ 為什麼這個 LS 對 FIC 是有效的？
  影像具有空間連續性（spatial locality of self-similarity）：
  如果某個 domain block D_j 跟 range block R_i 很相似，
  那 D_j 附近的 domain blocks 也很可能跟 R_i 相似。

  PPSO 找到的最佳解可能只是「附近區域的某個還行的解」，
  Spatial LS 透過檢查空間鄰居，把這個「附近」掃乾淨。

■ 鄰域定義：
  Domain block 在原圖中有一個空間位置 (x, y)。
  Spatial neighbors 是上下左右斜方向，偏移一個 stride 的 domain blocks。

           NW   N   NE
             ↖ ↑ ↗
        W ← (x,y) → E
             ↙ ↓ ↘
           SW   S   SE

  共 8 個鄰居（不含自己）。對每個鄰居：
  - 用相同的 isometry 做 fitness evaluation
  - 也可以對所有 isometry 都試 → 更徹底但更慢

■ Time complexity：
  每次 LS 最多 8 (鄰居) × 1 或 8 (isometry) 次 fitness evaluation
  我們採用「fix isometry」策略：8 次 evaluation，速度與效果的平衡。

■ 邊界處理：
  若鄰居超出影像邊界（例如 domain 在角落），跳過該方向。
=============================================================================
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


# 8 個方向的偏移量（以「stride」為單位）
# 例如 stride=8 時，偏移就是 (-8, 0), (8, 0) 等等
SPATIAL_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def build_domain_index_lookup(domain_positions, image_shape,
                              domain_size, domain_stride):
    """
    建立從 (row, col) 座標到 domain_idx 的反向查找表。
    讓我們可以「給座標 → 拿到對應的 domain index」。

    Returns:
        position_to_idx: dict {(row, col): domain_idx}
    """
    return {pos: idx for idx, pos in enumerate(domain_positions)}


def spatial_ls(range_block, all_iso, current_solution, n_domain,
               domain_positions, position_to_idx,
               domain_stride=8, try_all_isometries=False, **kwargs):
    """
    Spatial Neighborhood Local Search。

    Args:
        range_block:        當前 range block (8x8)
        all_iso:            預計算的 isometry 查表
        current_solution:   dict {'d_idx', 'iso', 's', 'o', 'mse'}
        n_domain:           domain pool 大小
        domain_positions:   list of (row, col), domain blocks 的位置
        position_to_idx:    dict, 從位置反查 index 的字典
        domain_stride:      domain 的 stride（決定鄰居距離）
        try_all_isometries: 若 True，每個鄰居都試 8 種 isometry
                            若 False（預設），只用當前 isometry

    Returns:
        improved_solution: dict (若找到更好) 或 current_solution
        n_evaluations:     此次 LS 用掉的 evaluation 次數
    """
    best = dict(current_solution)
    n_evals = 0

    # 取出當前 domain block 的空間座標
    cur_d_idx = current_solution['d_idx']
    cur_row, cur_col = domain_positions[cur_d_idx]
    cur_iso = current_solution['iso']

    # 走訪 8 個方向
    for dr, dc in SPATIAL_OFFSETS_8:
        neighbor_row = cur_row + dr * domain_stride
        neighbor_col = cur_col + dc * domain_stride

        # 查詢這個座標是否存在合法的 domain block
        neighbor_idx = position_to_idx.get((neighbor_row, neighbor_col))
        if neighbor_idx is None:
            continue  # 邊界外，跳過

        # 評估鄰居
        if try_all_isometries:
            iso_range = range(8)
        else:
            iso_range = [cur_iso]

        for iso in iso_range:
            s, o, mse = core.evaluate_candidate(
                range_block, all_iso, neighbor_idx, iso
            )
            n_evals += 1
            if abs(s) >= 1.0:
                continue
            if mse < best['mse']:
                best = {
                    'd_idx': neighbor_idx,
                    'iso': iso,
                    's': float(s),
                    'o': float(o),
                    'mse': float(mse),
                }

    return best, n_evals
