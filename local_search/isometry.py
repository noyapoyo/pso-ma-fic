"""
=============================================================================
local_search/isometry.py - Isometry Cycling Local Search (C2)
=============================================================================

■ 核心 idea：
  固定 domain block，遍歷其他 7 種 isometry 變換，找最佳變換。

■ 為什麼這個 LS 對 FIC 是有效的？
  PPSO 在 (domain_idx, isometry) 二維空間中搜索，但 isometry 只有 8 個
  離散值，PSO 的連續搜索 + round 解碼有時會「跳過」某些 isometry。
  Isometry LS 用窮舉的方式檢查所有 isometry，確保不會錯過最佳變換。

  另一個直觀理解：當 PPSO 已經找到一個「不錯的 domain」，
  通常只剩「最適合的 isometry」沒選對，這個 LS 就是專門修這個。

■ Time complexity：
  最多 7 次 fitness evaluation（不重複當前 isometry）。
  這是所有 LS 算子中最便宜的！

■ 使用時機：
  通常作為「第一道精煉」，因為成本低、效果好。
  在 LocalSearchPipeline 中可以放在第一個。
=============================================================================
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


def isometry_ls(range_block, all_iso, current_solution,
                n_domain=None, **kwargs):
    """
    Isometry Cycling Local Search。

    固定 domain_idx，遍歷其他 7 種 isometry 找最佳。

    Args:
        range_block:      當前 range block
        all_iso:          預計算的 isometry 查表
        current_solution: dict {'d_idx', 'iso', 's', 'o', 'mse'}
        n_domain:         (未使用，介面相容)

    Returns:
        improved_solution: dict
        n_evaluations:     1~7
    """
    best = dict(current_solution)
    n_evals = 0

    cur_d_idx = current_solution['d_idx']
    cur_iso = current_solution['iso']

    for iso in range(8):
        if iso == cur_iso:
            continue  # 跳過當前 isometry（已知 fitness）

        s, o, mse = core.evaluate_candidate(
            range_block, all_iso, cur_d_idx, iso
        )
        n_evals += 1
        if abs(s) >= 1.0:
            continue
        if mse < best['mse']:
            best = {
                'd_idx': cur_d_idx,
                'iso': iso,
                's': float(s),
                'o': float(o),
                'mse': float(mse),
            }

    return best, n_evals
