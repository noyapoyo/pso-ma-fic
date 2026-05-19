"""
=============================================================================
fic_budget.py - Fitness Function Evaluation (FFE) Budget Tracker
=============================================================================

提供統一的 FFE 預算追蹤機制，確保不同 metaheuristic 在公平的計算預算下比較。

■ 為什麼需要 FFE budget？
  比較不同 metaheuristic (PSO, PPSO, GA, Memetic-*) 時，常見的不公平因素：
    - 「相同 iteration」不公平：每代消耗的 FFE 不同
                                 (PSO 每代 40 FFE vs PPSO 每代 28 FFE)
    - 「相同時間」不公平：受 implementation 效率影響
    - 「相同 FFE」公平：直接比較演算法在「相同計算努力」下的能力

  FFE 是 metaheuristic 文獻公認的公平比較標準。

■ 使用方式：

  # 1. 建立 budget
  budget = FFEBudget(max_evals=1500)

  # 2. 每次評估前檢查 + 增加計數
  while not budget.exhausted():
      ...
      s, o, mse = core.evaluate_candidate(...)
      budget.consume(1)

  # 3. 查詢剩餘預算
  remaining = budget.remaining()
=============================================================================
"""


class FFEBudget:
    """
    追蹤 FFE 消耗，當預算用盡時通知 encoder 停止搜索。

    Attributes:
        max_evals: 預算上限
        used:      已消耗的 FFE 數
    """

    def __init__(self, max_evals=None):
        """
        Args:
            max_evals: 最大 FFE 數，None 表示無限制（用於 Full Search）
        """
        self.max_evals = max_evals
        self.used = 0

    def consume(self, n=1):
        """消耗 n 個 FFE。Return True 若還有預算可用。"""
        self.used += n
        return not self.exhausted()

    def exhausted(self):
        """預算是否用盡。"""
        if self.max_evals is None:
            return False
        return self.used >= self.max_evals

    def remaining(self):
        """剩餘預算。"""
        if self.max_evals is None:
            return float('inf')
        return max(0, self.max_evals - self.used)

    def reset(self):
        """重置計數器，準備處理下一個 range block。"""
        self.used = 0

    def __repr__(self):
        if self.max_evals is None:
            return f"FFEBudget(used={self.used}, no limit)"
        pct = 100 * self.used / self.max_evals
        return (f"FFEBudget(used={self.used}/{self.max_evals}, "
                f"{pct:.1f}%)")
