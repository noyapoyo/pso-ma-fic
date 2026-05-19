"""
=============================================================================
fic_core.py - Fractal Image Compression 共用核心模組
=============================================================================

本模組包含所有 FIC encoder 共用的元件。任何新方法 (PSO, PPSO, Memetic, ...)
只需 import 此模組，並實作自己的 encode 函式即可。

■ 共用組件 (不需要重寫)：
    - 影像分割 (range / domain blocks)
    - Isometry 變換 (8 種對稱)
    - Affine 參數計算 (s, o 的最小二乘解) ← 也是 fitness function 的核心
    - Decoding (Banach 不動點迭代)
    - PSNR 計算
    - I/O 與 pipeline (run_pipeline)

■ 設計原則：
    - 所有 encoder 共用相同 fitness 評估邏輯 → 公平比較
    - 所有 encoder 共用相同 decoder → 結果格式一致
    - encoder 之間的差異只在「搜索策略」(search strategy)
=============================================================================
"""

import numpy as np
from PIL import Image
import time
import os
import json


# =============================================================================
# Part 1: 影像分割 (Image Partitioning)
# =============================================================================

def extract_range_blocks(image, block_size=8):
    """
    將影像切成不重疊的 range blocks。

    Returns:
        blocks: list of 2D float64 arrays, shape=(block_size, block_size)
        positions: list of (row, col) 左上角座標
    """
    h, w = image.shape
    blocks, positions = [], []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape == (block_size, block_size):
                blocks.append(block.astype(np.float64))
                positions.append((i, j))
    return blocks, positions


def extract_domain_blocks(image, domain_size=16, stride=8, range_size=8):
    """
    提取 domain blocks 並 downsample (2x2 average pooling) 到 range_size。

    stride 決定 domain pool 大小：
      stride=8  -> 31² = 961 blocks
      stride=4  -> 61² = 3721 blocks (更密但更慢)
    """
    h, w = image.shape
    blocks, positions = [], []
    scale = domain_size // range_size

    for i in range(0, h - domain_size + 1, stride):
        for j in range(0, w - domain_size + 1, stride):
            block = image[i:i + domain_size, j:j + domain_size].astype(np.float64)
            downsampled = block.reshape(
                range_size, scale, range_size, scale
            ).mean(axis=(1, 3))
            blocks.append(downsampled)
            positions.append((i, j))
    return blocks, positions


# =============================================================================
# Part 2: Isometry 變換 (8 種等距變換)
# =============================================================================

def apply_isometry(block, iso_type):
    """8 種 Dihedral group 變換。"""
    if iso_type == 0:   return block.copy()
    elif iso_type == 1: return np.rot90(block, 1)
    elif iso_type == 2: return np.rot90(block, 2)
    elif iso_type == 3: return np.rot90(block, 3)
    elif iso_type == 4: return np.fliplr(block)
    elif iso_type == 5: return np.flipud(block)
    elif iso_type == 6: return block.T
    elif iso_type == 7: return np.rot90(block, 1).T
    else: raise ValueError(f"Invalid isometry: {iso_type}")


def precompute_all_isometries(domain_blocks):
    """
    預計算所有 domain blocks × 8 種 isometry 的查表。
    Returns: array shape=(n_domains, 8, block_size, block_size)
    所有 encoder 都共用這個查表 (避免重複計算)。
    """
    n = len(domain_blocks)
    bs = domain_blocks[0].shape[0]
    result = np.zeros((n, 8, bs, bs))
    for d_idx in range(n):
        for iso in range(8):
            result[d_idx, iso] = apply_isometry(domain_blocks[d_idx], iso)
    return result


# =============================================================================
# Part 3: Fitness Function (所有 encoder 的核心)
# =============================================================================
#
# 給定 (range_block, domain_block, isometry)，計算最優 (s, o) 與對應 MSE。
# 這是所有 encoder 共用的「目標函數」評估邏輯。
#
# Full Search:    對每個 (d, k) 組合都呼叫一次
# PSO/PPSO:       每個粒子的 fitness 評估呼叫一次
# Memetic:        global search + local search 都呼叫
# =============================================================================

def compute_affine_params(range_block, domain_transformed):
    """
    最小二乘解析解：給定 R 和 T_k(D)，求最優 contrast s 和 brightness o。

    s = [n·Σ(d·r) - Σd·Σr] / [n·Σ(d²) - (Σd)²]
    o = [Σr - s·Σd] / n
    MSE = (1/n) Σ(r - s·d - o)²

    Returns: (s, o, mse)
    """
    r = range_block.ravel()
    d = domain_transformed.ravel()
    n = len(r)

    sum_d = np.sum(d)
    sum_r = np.sum(r)
    sum_dd = np.sum(d * d)
    sum_dr = np.sum(d * r)

    denom = n * sum_dd - sum_d * sum_d

    if abs(denom) < 1e-10:
        # Domain block 近乎純色：s 無法確定
        s = 0.0
        o = sum_r / n
    else:
        s = (n * sum_dr - sum_d * sum_r) / denom
        o = (sum_r - s * sum_d) / n

    diff = r - s * d - o
    mse = np.dot(diff, diff) / n
    return s, o, mse


def evaluate_candidate(range_block, all_iso, d_idx, iso):
    """
    Encoder 統一的 fitness 評估介面。
    輸入：range_block, isometry 查表, domain index, isometry index
    輸出：(s, o, mse)

    所有方法 (Full Search, PSO, PPSO, Memetic) 都呼叫此函式評估候選解。
    """
    return compute_affine_params(range_block, all_iso[d_idx, iso])


# =============================================================================
# Part 4: Decoding (Banach 不動點迭代)
# =============================================================================
#
# 從 fractal codes 重建影像。
# 所有 encoder 產出的 fractal_codes 格式相同 → 共用 decoder。
# =============================================================================

def decode(fractal_codes, image_shape,
           range_size=8, domain_size=16, n_iterations=20, verbose=True):
    """
    從 fractal codes 重建影像。

    fractal_codes 中每個 entry 必須包含：
      'range_pos', 'domain_pos', 'isometry', 'contrast', 'brightness'
    """
    h, w = image_shape
    scale = domain_size // range_size
    current = np.full((h, w), 128.0)  # 初始全灰影像

    if verbose:
        print(f"  Decoding ({n_iterations} iterations)...", end=" ", flush=True)

    for _ in range(n_iterations):
        next_img = np.zeros_like(current)
        for code in fractal_codes:
            r_row, r_col = code['range_pos']
            d_row, d_col = code['domain_pos']

            # 從當前影像取 domain block + downsample
            d_block = current[d_row:d_row + domain_size,
                              d_col:d_col + domain_size]
            d_down = d_block.reshape(
                range_size, scale, range_size, scale
            ).mean(axis=(1, 3))

            # 套用 isometry + affine transformation
            d_trans = apply_isometry(d_down, code['isometry'])
            new_block = code['contrast'] * d_trans + code['brightness']

            next_img[r_row:r_row + range_size,
                     r_col:r_col + range_size] = new_block

        current = np.clip(next_img, 0, 255)

    if verbose:
        print("done")
    return current.astype(np.uint8)


# =============================================================================
# Part 5: 評估與 I/O 工具
# =============================================================================

def compute_psnr(original, reconstructed):
    """計算 PSNR (dB)。"""
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    mse = np.mean((o - r) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def load_image_as_gray(image_path, image_size=256):
    """讀圖 → 灰階 → image_size × image_size (LANCZOS resize)。"""
    img = Image.open(image_path).convert('L')
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.LANCZOS)
    return np.array(img)


# 向後相容：保留舊名稱
def load_image_as_gray256(image_path):
    """[Deprecated] 用 load_image_as_gray(path, 256) 代替。"""
    return load_image_as_gray(image_path, 256)


def save_codes_and_stats(output_dir, image_name, method_name, stats, fractal_codes):
    """儲存 stats.json 和 codes.json。"""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{image_name}_{method_name}"

    stats_path = os.path.join(output_dir, f"{prefix}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    codes_serializable = []
    for c in fractal_codes:
        codes_serializable.append({
            'range_pos': list(c['range_pos']),
            'domain_idx': int(c['domain_idx']),
            'domain_pos': list(c['domain_pos']),
            'isometry': int(c['isometry']),
            'contrast': float(c['contrast']),
            'brightness': float(c['brightness']),
            'mse': float(c['mse']),
        })
    codes_path = os.path.join(output_dir, f"{prefix}_codes.json")
    with open(codes_path, 'w') as f:
        json.dump(codes_serializable, f)


def estimate_compression_ratio(image_shape, n_range, n_domain):
    """
    估算壓縮比。每個 fractal code 編碼成本：
      domain_idx: ceil(log2(n_domain)) bits
      isometry:   3 bits (8 種)
      contrast s: 8 bits (量化)
      brightness o: 8 bits (量化)
    """
    bits_per_code = int(np.ceil(np.log2(n_domain))) + 3 + 8 + 8
    total_code_bits = n_range * bits_per_code
    original_bits = image_shape[0] * image_shape[1] * 8
    return original_bits / total_code_bits, bits_per_code


# =============================================================================
# Part 6: 統一的 Pipeline (run_pipeline)
# =============================================================================
#
# 任何 encoder 都可以套用此 pipeline。Encoder 只需符合介面：
#
#   encode_fn(image, range_size, domain_size, domain_stride, **kwargs)
#       -> (fractal_codes, encoding_time, stats, domain_positions)
# =============================================================================

def run_pipeline(encode_fn, image_path, method_name,
                 output_dir="results",
                 image_size=256,
                 range_size=8, domain_size=16, domain_stride=8,
                 decode_iterations=20, save_outputs=True,
                 save_fic=True,
                 **encoder_kwargs):
    """
    通用 pipeline: load → encode → decode → evaluate → save

    Args:
        encode_fn:    encoder 函式
        image_path:   影像路徑
        method_name:  方法名稱
        image_size:   讀圖後 resize 的目標邊長 (預設 256)
        range_size, domain_size, domain_stride: FIC 分割參數
        encoder_kwargs: 傳給 encoder 的額外參數
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = load_image_as_gray(image_path, image_size)

    print(f"\n{'#' * 64}")
    print(f"  [{method_name.upper()}]  Processing: {image_name}")
    print(f"  Image: {image.shape}, range=[{image.min()}, {image.max()}]")
    if encoder_kwargs:
        print(f"  Encoder params: {encoder_kwargs}")
    print(f"{'#' * 64}\n")

    # --- Encoding ---
    fractal_codes, enc_time, stats, _ = encode_fn(
        image,
        range_size=range_size,
        domain_size=domain_size,
        domain_stride=domain_stride,
        **encoder_kwargs
    )

    # --- Decoding ---
    reconstructed = decode(
        fractal_codes, image.shape,
        range_size, domain_size, decode_iterations
    )

    # --- 評估 ---
    psnr_actual = compute_psnr(image, reconstructed)
    cr, bits_per_code = estimate_compression_ratio(
        image.shape, stats['n_range'], stats['n_domain']
    )

    stats['method'] = method_name
    stats['image'] = image_name
    stats['psnr_actual'] = round(psnr_actual, 2)
    stats['compression_ratio'] = round(cr, 2)
    stats['bits_per_code'] = bits_per_code
    stats['decode_iterations'] = decode_iterations

    # --- 儲存 ---
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(image).save(
            os.path.join(output_dir, f"{image_name}_original.png"))
        Image.fromarray(reconstructed).save(
            os.path.join(output_dir, f"{image_name}_{method_name}_reconstructed.png"))

        diff = np.abs(image.astype(np.float64) - reconstructed.astype(np.float64))
        Image.fromarray(np.clip(diff * 5, 0, 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{image_name}_{method_name}_diff_x5.png"))

        save_codes_and_stats(output_dir, image_name, method_name, stats, fractal_codes)

        # --- 儲存 .fic 二進制壓縮檔 ---
        if save_fic:
            from fic_bitstream import save_fic as _save_fic
            fic_path = os.path.join(output_dir,
                                    f"{image_name}_{method_name}.fic")
            fic_size = _save_fic(
                fic_path, fractal_codes, image.shape,
                range_size=range_size, domain_size=domain_size,
                domain_stride=domain_stride, n_domain=stats['n_domain'],
            )
            stats['fic_file_size_bytes'] = fic_size
            stats['fic_file_size_kb'] = round(fic_size / 1024, 2)

    print(f"\n  === SUMMARY: {image_name} ({method_name}) ===")
    print(f"    Encoding time:     {enc_time:.2f} sec")
    print(f"    PSNR:              {psnr_actual:.2f} dB")
    print(f"    Compression ratio: {cr:.1f}:1")
    print(f"    Avg MSE:           {stats['mse_mean']:.4f}")
    if 'n_evaluations' in stats:
        print(f"    Fitness evals:     {stats['n_evaluations']:,}")
    if 'fic_file_size_kb' in stats:
        print(f"    .fic file size:    {stats['fic_file_size_kb']} KB  "
              f"(original: {image.shape[0]*image.shape[1]//1024} KB)")
    print()

    return stats


def print_summary_table(all_stats):
    """印出多筆實驗結果的比較表。"""
    print("\n" + "=" * 96)
    print("  RESULTS SUMMARY")
    print("=" * 96)
    print(f"  {'Image':<18} {'Method':<16} {'Time(s)':>10} "
          f"{'PSNR(dB)':>9} {'CR':>8} {'AvgMSE':>10} {'Evals':>14}")
    print("-" * 96)
    for s in all_stats:
        evals = f"{s.get('n_evaluations', 0):,}" if 'n_evaluations' in s else "-"
        print(f"  {s['image']:<18} {s['method']:<16} "
              f"{s['encoding_time_sec']:>10.2f} "
              f"{s['psnr_actual']:>9.2f} "
              f"{s['compression_ratio']:>7.1f}:1 "
              f"{s['mse_mean']:>10.4f} "
              f"{evals:>14}")
    print("=" * 96)
