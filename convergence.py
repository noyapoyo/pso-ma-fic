"""
=============================================================================
convergence.py - Convergence Curve 記錄與聚合工具
=============================================================================

用於記錄每個 range block 在搜尋過程中 (FFE, best_MSE) 的變化，
並聚合成整張影像的 convergence curve (FFE → avg PSNR)。

■ 用法 (在 encoder 的 *_search_one_range 中):

    log = []   # 每個 block 一個 list
    # 初始化後
    log.append((n_evals, gbest_mse))
    # 每個 iteration 後
    log.append((n_evals, gbest_mse))
    # 回傳 log

■ 用法 (在 encoder 主函式中):

    all_logs = []
    for r_idx in range(n_range):
        best, stats, log = search_one_range(...)
        all_logs.append(log)
    curve = aggregate_convergence(all_logs, budget=500, n_points=50)
    # curve = [(ffe, avg_psnr), ...]

=============================================================================
"""

import numpy as np
import json
import os


def aggregate_convergence(all_block_logs, budget, n_points=50):
    """
    將所有 range block 的 convergence log 聚合成一條 curve。

    Args:
        all_block_logs: list of list of (ffe, best_mse)
                        每個 block 一個 list，內含 (累計FFE, 當時gbest_mse) snapshots
        budget:         FFE budget per block
        n_points:       輸出幾個 checkpoint

    Returns:
        list of dict: [{'ffe': int, 'avg_mse': float, 'psnr_db': float}, ...]
    """
    if not all_block_logs:
        return []

    # 定義統一的 FFE checkpoints
    checkpoints = np.linspace(0, budget, n_points + 1, dtype=int)
    checkpoints = sorted(set(checkpoints))  # 去重

    n_blocks = len(all_block_logs)
    curve = []

    for cp in checkpoints:
        mse_at_cp = []
        for log in all_block_logs:
            if not log:
                continue
            # 找 log 中 ffe <= cp 的最後一筆（即該 checkpoint 時的 best MSE）
            best_mse = log[0][1]  # fallback: 初始值
            for ffe, mse in log:
                if ffe <= cp:
                    best_mse = mse
                else:
                    break
            mse_at_cp.append(best_mse)

        if not mse_at_cp:
            continue

        avg_mse = float(np.mean(mse_at_cp))
        psnr = 10 * np.log10(255.0 ** 2 / avg_mse) if avg_mse > 0 else float('inf')
        curve.append({
            'ffe': int(cp),
            'avg_mse': round(avg_mse, 4),
            'psnr_db': round(float(psnr), 4),
        })

    return curve


def save_convergence(curves_by_method, output_dir, image_name, run_idx=0):
    """
    儲存 convergence data 為 JSON。

    Args:
        curves_by_method: dict {method_name: [{'ffe', 'avg_mse', 'psnr_db'}, ...]}
        output_dir:       輸出目錄
        image_name:       影像名稱
        run_idx:          run index
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir,
                        f"convergence_{image_name}_run{run_idx}.json")
    with open(path, 'w') as f:
        json.dump(curves_by_method, f, indent=2)
    return path
