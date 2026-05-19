"""
=============================================================================
local_search package - Local Search 算子集合
=============================================================================

本套件提供針對 FIC 問題特性設計的 local search 算子。
每個算子都遵循統一介面，可以單獨使用或組合使用。

■ 介面定義 (所有 LS 算子都實作以下函式)：

    def search(range_block, all_iso, current_solution, n_domain,
               domain_positions=None, **kwargs)
        -> (improved_solution, n_evaluations)

  其中 current_solution 是 dict:
      {'d_idx': int, 'iso': int, 's': float, 'o': float, 'mse': float}

  improved_solution 是同樣格式的 dict（若沒找到更好就回傳原解）

■ 已實作的算子：
    - SpatialLS:  C1 - 空間鄰域搜索
    - IsometryLS: C2 - Isometry 遍歷搜索

■ 設計原則：
    - 各算子獨立，可單獨啟用/停用 → 方便做 ablation study
    - 統一介面 → 可用 LocalSearchPipeline 組合多個算子
    - 都共用 fic_core.evaluate_candidate 做 fitness 評估
=============================================================================
"""

from .spatial import spatial_ls
from .isometry import isometry_ls
from .pipeline import LocalSearchPipeline

__all__ = ['spatial_ls', 'isometry_ls', 'LocalSearchPipeline']
