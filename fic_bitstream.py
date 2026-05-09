"""
Save fractal codes to bianry compression file (.fic).

Each fractal code's bit layout(default 29 bits)：
  bits 28..19 : domain_idx  (10 bits, at most 1024 domain blocks)
  bits 18..16 : isometry    ( 3 bits)
  bits 15.. 8 : contrast    ( 8 bits)
  bits  7.. 0 : brightness  ( 8 bits)
"""

import numpy as np
import struct
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fic_core as core


# contrast s: [-1.0, 1.0]
S_MIN, S_MAX = -1.0, 1.0

# brightness o: Actual range is [-350, 350]，but we use [-512, 512] to totally cover 
O_MIN, O_MAX = -512.0, 512.0


def quantize(value, v_min, v_max, n_bits):
    """float to integer [0, 2^{n_bits} - 1]。"""
    levels = (1 << n_bits) - 1          # 2^{n_bits} - 1
    q = round((value - v_min) / (v_max - v_min) * levels)
    return int(np.clip(q, 0, levels))


def dequantize(q, v_min, v_max, n_bits):
    """integer to float"""
    levels = (1 << n_bits) - 1
    return v_min + q / levels * (v_max - v_min)

def pack_codes_to_bytes(fractal_codes, n_domain, bits_s=8, bits_o=8):
    """
    Pack fractal codes tightly, and padding with zeros if the last byte is less than 1 byte.

    Each code bit layout：
      [domain_idx (bits_d bits) | isometry (3 bits) | s_q (bits_s) | o_q (bits_o)]

    Returns:
        payload: bytes
        bits_per_code: int
        n_domain: int
    """
    bits_d = int(np.ceil(np.log2(max(n_domain, 2))))
    bits_per_code = bits_d + 3 + bits_s + bits_o

    # code bitstring
    total_bits = len(fractal_codes) * bits_per_code
    total_bytes = (total_bits + 7) // 8  # ceiling

    payload = bytearray(total_bytes)
    bit_pos = 0

    for code in fractal_codes:
        d_idx = int(np.clip(code['domain_idx'], 0, n_domain - 1))
        iso   = int(np.clip(code['isometry'], 0, 7))
        s_q   = quantize(code['contrast'],    S_MIN, S_MAX, bits_s)
        o_q   = quantize(code['brightness'],  O_MIN, O_MAX, bits_o)

        # MSB first
        code_int = (d_idx << (3 + bits_s + bits_o)) | \
                   (iso   << (bits_s + bits_o))     | \
                   (s_q   << bits_o)                | \
                   o_q

        # Write payload per bit
        for bit_offset in range(bits_per_code - 1, -1, -1):
            if (code_int >> bit_offset) & 1:
                byte_idx = bit_pos // 8
                bit_in_byte = 7 - (bit_pos % 8)
                payload[byte_idx] |= (1 << bit_in_byte)
            bit_pos += 1

    return bytes(payload), bits_per_code


def unpack_bytes_to_codes(payload, n_codes, domain_positions,
                          range_positions, n_domain,
                          bits_d, bits_s=8, bits_o=8):
    """
    Unpack byte array to fractal codes list.

    Returns:
        fractal_codes: list of dicts (Same with encoder format)
    """
    bits_per_code = bits_d + 3 + bits_s + bits_o
    mask_d   = (1 << bits_d) - 1
    mask_iso = 0b111
    mask_s   = (1 << bits_s) - 1
    mask_o   = (1 << bits_o) - 1

    fractal_codes = []
    bit_pos = 0

    for r_idx in range(n_codes):
        # Read bits_per_code bits
        code_int = 0
        for _ in range(bits_per_code):
            byte_idx = bit_pos // 8
            bit_in_byte = 7 - (bit_pos % 8)
            bit_val = (payload[byte_idx] >> bit_in_byte) & 1
            code_int = (code_int << 1) | bit_val
            bit_pos += 1

        # decode 
        o_q   = code_int & mask_o;          code_int >>= bits_o
        s_q   = code_int & mask_s;          code_int >>= bits_s
        iso   = code_int & mask_iso;        code_int >>= 3
        d_idx = code_int & mask_d

        d_idx = int(np.clip(d_idx, 0, n_domain - 1))

        fractal_codes.append({
            'range_pos':  range_positions[r_idx],
            'domain_idx': d_idx,
            'domain_pos': domain_positions[d_idx],
            'isometry':   iso,
            'contrast':   dequantize(s_q, S_MIN, S_MAX, bits_s),
            'brightness': dequantize(o_q, O_MIN, O_MAX, bits_o),
            'mse':        0.0,  # Compression do not need to save MSE
        })

    return fractal_codes


# .fic file format: Header
MAGIC = b'FIC1'
HEADER_FORMAT = '>4s HH BBB BB 3s'   # big-endian
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)   # = 16 bytes


def build_header(img_height, img_width, range_size, domain_size,
                 domain_stride, bits_s, bits_o):
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        img_height, img_width,
        range_size, domain_size, domain_stride,
        bits_s, bits_o,
        b'\x00\x00\x00',
    )


def parse_header(data):
    fields = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    magic, h, w, rs, ds, dstride, bs, bo, _ = fields
    assert magic == MAGIC, f"Invalid magic: {magic}"
    return {
        'img_height':    h,
        'img_width':     w,
        'range_size':    rs,
        'domain_size':   ds,
        'domain_stride': dstride,
        'bits_s':        bs,
        'bits_o':        bo,
    }


# Public Interface
def save_fic(filepath, fractal_codes, image_shape,
             range_size=8, domain_size=16, domain_stride=8,
             bits_s=8, bits_o=12, n_domain=None):
    """
    Save fractal codes to .fic binary compression file.

    Args:
        filepath:      Output path
        fractal_codes: encoder list of dicts output
        image_shape:   (H, W)
        bits_s:        contrast (default 8)
        bits_o:        brightness (default 8)
        n_domain:      domain pool
    """
    h, w = image_shape

    if n_domain is None:
        n_domain = max(c['domain_idx'] for c in fractal_codes) + 1

    # pack bitstream
    payload, bits_per_code = pack_codes_to_bytes(
        fractal_codes, n_domain, bits_s, bits_o
    )

    header = build_header(h, w, range_size, domain_size,
                          domain_stride, bits_s, bits_o)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(payload)

    file_size = os.path.getsize(filepath)
    original_bits = h * w * 8
    cr = original_bits / (len(fractal_codes) * bits_per_code)

    print(f"  Saved: {filepath}")
    print(f"    Header:    {HEADER_SIZE} bytes")
    print(f"    Payload:   {len(payload)} bytes  "
          f"({bits_per_code} bits × {len(fractal_codes)} codes)")
    print(f"    Total:     {file_size} bytes  ({file_size/1024:.2f} KB)")
    print(f"    Original:  {original_bits//8} bytes  "
          f"({original_bits//8//1024} KB)")
    print(f"    Ratio:     {cr:.2f}:1  "
          f"({len(fractal_codes) * bits_per_code / (h*w):.3f} bpp)")

    return file_size


def load_fic(filepath, verbose=True):
    """
    load fractal codes from .fic.

    Returns:
        fractal_codes: list of dicts
        meta: dict with image_shape, range_size, image_size, domain_size, domain_stride
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    header = parse_header(raw)
    payload = raw[HEADER_SIZE:]

    h               = header['img_height']
    w               = header['img_width']
    range_size      = header['range_size']
    domain_size     = header['domain_size']
    domain_stride   = header['domain_stride']
    bits_s          = header['bits_s']
    bits_o          = header['bits_o']

    if verbose:
        print(f"  Loaded: {filepath}")
        print(f"    Image: {h}x{w}, range={range_size}, "
              f"domain={domain_size}, stride={domain_stride}")

    # Reconstruct range / domain position index 
    dummy = np.zeros((h, w), dtype=np.uint8)
    _, range_positions = core.extract_range_blocks(dummy, range_size)
    _, domain_positions = core.extract_domain_blocks(
        dummy, domain_size, domain_stride, range_size
    )

    n_codes  = len(range_positions)
    n_domain = len(domain_positions)
    bits_d   = int(np.ceil(np.log2(max(n_domain, 2))))

    fractal_codes = unpack_bytes_to_codes(
        payload, n_codes, domain_positions, range_positions,
        n_domain, bits_d, bits_s, bits_o
    )

    meta = {
        'image_shape':      (h, w),
        'range_size':       range_size,
        'image_size':       h * w,
        'domain_size':      domain_size,
        'domain_stride':    domain_stride,
    }

    return fractal_codes, meta

# Decode from .fic file
def decode_fic_file(fic_path, output_path=None, decode_iterations=20):
    """
    Reconstruct image from .fic file.
    """
    from PIL import Image

    print(f"\nDecoding: {fic_path}")
    fractal_codes, meta = load_fic(fic_path)

    reconstructed = core.decode(
        fractal_codes,
        meta['image_shape'],
        image_size=meta['image_size'],
        range_size=meta['range_size'],
        domain_size=meta['domain_size'],
        n_iterations=decode_iterations,
    )

    if output_path is None:
        base = os.path.splitext(fic_path)[0]
        output_path = base + '_decoded.png'

    Image.fromarray(reconstructed).save(output_path)
    print(f"  Saved decoded image: {output_path}")
    return reconstructed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fic_bitstream.py <file.fic> [output.png]")
        sys.exit(1)
    fic_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    decode_fic_file(fic_path, out_path)
