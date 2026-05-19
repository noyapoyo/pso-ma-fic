"""
=============================================================================
local_search/pipeline.py - Local Search 組合器
=============================================================================

讓多個 LS 算子可以依序串接運作，類似 sklearn 的 Pipeline。

用法：
    pipeline = LocalSearchPipeline([
        ('isometry', isometry_ls),
        ('spatial', spatial_ls),
    ])
    improved, n_evals = pipeline.apply(range_block, all_iso, current, context)

Pipeline 會依序套用每個 LS 算子：
    current → isometry_ls → improved1 → spatial_ls → improved2

每個算子都拿前一個的輸出當作起點，這樣形成「層層精煉」的效果。

■ 為什麼這樣設計？
  - 統一介面：所有 LS 算子都用相同的 (search, n_evals) 回傳格式
  - 可組合：可以隨時調換順序、加減算子
  - 方便 ablation：在 encoder 那邊只要改 pipeline 內容就能換不同 LS 組合
=============================================================================
"""


class LocalSearchPipeline:
    """
    依序串接多個 LS 算子的容器。

    Attributes:
        operators: list of (name, function)
    """

    def __init__(self, operators=None):
        """
        Args:
            operators: list of (name, function)
                       例：[('isometry', isometry_ls), ('spatial', spatial_ls)]
        """
        self.operators = operators or []

    def add(self, name, func):
        """加入一個 LS 算子。"""
        self.operators.append((name, func))
        return self

    def apply(self, range_block, all_iso, current_solution, context):
        """
        依序套用所有 LS 算子。

        Args:
            range_block:      當前 range block
            all_iso:          isometry 查表
            current_solution: 起始解 dict
            context:          dict, 傳給各 LS 算子的額外參數
                              例：{'n_domain': 961, 'domain_positions': [...],
                                   'position_to_idx': {...}, 'domain_stride': 8}

        Returns:
            best_solution: dict
            total_evals:   整個 pipeline 用掉的 evaluation 次數
            per_op_evals:  dict 各算子各自用掉的 evaluations
        """
        best = current_solution
        total_evals = 0
        per_op_evals = {}

        for name, func in self.operators:
            improved, n_evals = func(
                range_block, all_iso, best, **context
            )
            total_evals += n_evals
            per_op_evals[name] = n_evals
            best = improved

        return best, total_evals, per_op_evals

    def __repr__(self):
        names = [name for name, _ in self.operators]
        return f"LocalSearchPipeline({names})"

    def __len__(self):
        return len(self.operators)
