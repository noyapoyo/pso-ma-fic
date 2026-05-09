import numpy as np
from PIL import Image
import time
import os
import json

def extract_range_blocks(image, block_size=8):
    """
    Partition the image to non overlapped range blocks。

    Returns:
        blocks: list of 2D float64 arrays, shape=(block_size, block_size)
        positions: list of (row, col) top right coordinate.
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
    Extract domain blocks and downsample (2x2 average pooling) to range_size。

    stride determine the size of domain pool:
      stride=8  -> 31x**2 = 961 blocks
      stride=4  -> 61**2 = 3721 blocks
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


def apply_isometry(block, iso_type):
    """Dihedral transformation"""
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
    Precompute all domain blocks × 8 isometry lookup table.
    Returns: array shape=(n_domains, 8, block_size, block_size)
    All encoder will use this lookup table, prevent calculate repeatedly。
    """
    n = len(domain_blocks)
    bs = domain_blocks[0].shape[0]
    result = np.zeros((n, 8, bs, bs))
    for d_idx in range(n):
        for iso in range(8):
            result[d_idx, iso] = apply_isometry(domain_blocks[d_idx], iso)
    return result


# Affine trasformation parameters and Fitness function
def compute_affine_params(range_block, domain_transformed):
    """
    Compute affine trasformation parameters.
    By Least Squares Method to compute the optimal constrast s and brightness o.

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
    Input: (range_block, isometry lookup, domain index, isometry index)
    Wrapper fitness function for all method.

    Returns: (s, o, mse)
    """
    return compute_affine_params(range_block, all_iso[d_idx, iso])


# Decoding (Banach fixed point theorem)
def decode(fractal_codes, image_shape, image_size,
           range_size=8, domain_size=16, n_iterations=20, verbose=True):
    """
    Reconstuct image from fractal codes.
    Please ensure your encoder algorithm generate same fractal_codes with our format.

    fractal_codes entry foramt:
      'range_pos', 'domain_pos', 'isometry', 'contrast', 'brightness'
    """
    h, w = image_shape
    scale = domain_size // range_size
    current = np.full((h, w), image_size)  # Decode start with gray figure 

    if verbose:
        print(f"  Decoding ({n_iterations} iterations)...", end=" ", flush=True)

    for _ in range(n_iterations):
        next_img = np.zeros_like(current)
        for code in fractal_codes:
            r_row, r_col = code['range_pos']
            d_row, d_col = code['domain_pos']

            # domain block + downsample
            d_block = current[d_row:d_row + domain_size,
                              d_col:d_col + domain_size]
            d_down = d_block.reshape(
                range_size, scale, range_size, scale
            ).mean(axis=(1, 3))

            # isometry + affine transformation
            d_trans = apply_isometry(d_down, code['isometry'])
            new_block = code['contrast'] * d_trans + code['brightness']

            next_img[r_row:r_row + range_size,
                     r_col:r_col + range_size] = new_block

        current = np.clip(next_img, 0, 255)

    if verbose:
        print("done")
    return current.astype(np.uint8)


def compute_psnr(original, reconstructed):
    """Compute PSNR (dB)"""
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    mse = np.mean((o - r) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def load_image_as_gray(image_path, image_size):
    """Load the image and resize to `image_size` (LANCZOS resize)."""
    img = Image.open(image_path).convert('L')
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.LANCZOS)
    return np.array(img)


def save_codes_and_stats(output_dir, image_name, method_name, stats, fractal_codes):
    """Save stats.json and codes.json."""
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
    Compute the compression ratio, Every fractal code encode cost:
      domain_idx: ceil(log2(n_domain)) bits
      isometry:   3 bits (8 types)
      contrast s: 8 bits
      brightness o: 8 bits
    """
    bits_per_code = int(np.ceil(np.log2(n_domain))) + 3 + 8 + 8
    total_code_bits = n_range * bits_per_code
    original_bits = image_shape[0] * image_shape[1] * 8
    return original_bits / total_code_bits, bits_per_code


def run_pipeline(encode_fn, image_path, method_name,
                 image_size, range_size, domain_size,
                 output_dir="results",  domain_stride=8,
                 decode_iterations=20, save_outputs=True,
                 save_fic=True,
                 **encoder_kwargs):
    """
    General pipeline: load -> encode -> decode -> evaluate -> save

    Args:
        encode_fn: encoder function appropriate prototype 
            f(image, range_size, domain_size, domain_stride, **kwargs)
            -> (fractal_codes, encoding_time, stats, domain_positions)
        image_path: image path 
        method_name: function name 
        encoder_kwargs: other parameters (e.g. pop_size=40 for PSO)
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = load_image_as_gray(image_path, image_size)

    print(f"\n{'#' * 64}")
    print(f"  [{method_name.upper()}]  Processing: {image_name}")
    print(f"  Image: {image.shape}, range=[{image.min()}, {image.max()}]")
    if encoder_kwargs:
        print(f"  Encoder params: {encoder_kwargs}")
    print(f"{'#' * 64}\n")

    # Encoding
    fractal_codes, enc_time, stats, _ = encode_fn(
        image,
        range_size=range_size,
        domain_size=domain_size,
        domain_stride=domain_stride,
        **encoder_kwargs
    )

    # Decoding
    reconstructed = decode(
        fractal_codes, image.shape, image_size,
        range_size, domain_size, decode_iterations
    )

    # Evaluation
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

    # Save
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

        # Save .fic binary compression file
        if save_fic:
            from fic_bitstream import save_fic as _save_fic
            fic_path = os.path.join(output_dir, f"{image_name}_{method_name}.fic")

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
        print(f"    Fitness evals:      {stats['n_evaluations']:,}")
    if 'fic_file_size_kb' in stats:
        print(f"    .fic file size:     {stats['fic_file_size_kb']} KB"
              f"(original: {image.shape[0] * image.shape[1] // 1024} KB)")
    print()

    return stats


def print_summary_table(all_stats):
    """Print out the experiment result."""
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
