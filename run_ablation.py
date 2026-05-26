"""
=============================================================================
run_ablation.py - FG-PSO Ablation Study Runner
=============================================================================

Ablation studies to validate each component of FG-PSO:

1. Feature-Only Exhaustive (feature_exhaustive)
   - top-K × 8 exhaustive scan, no PSO/LS
   - Q: Is PSO even necessary after feature filtering?

2. FG-PSO without LS (fgpso_no_ls)
   - FG-PSO with ls_strategies=() (empty)
   - Q: Does memetic LS contribute?

3. FG-PSO with ISO-LS only (fgpso_iso_only)
   - FG-PSO with ls_strategies=('isometry',)
   - Q: Does Feature-Neighbor LS add value beyond Isometry LS?

4. K sensitivity (fgpso_k10, fgpso_k20, fgpso_k40, fgpso_k80, fgpso_k160)
   - FG-PSO with K = 10, 20, 40, 80, 160
   - Q: What is the optimal K?

5. FFE sensitivity (fgpso_b100, fgpso_b200, fgpso_b500, fgpso_b1000)
   - FG-PSO with B = 100, 200, 500, 1000
   - Q: How does FG-PSO perform at low budgets?

用法:
    # 跑全部 ablation (所有影像, 5 runs)
    python run_ablation.py --n-runs 5

    # 只跑特定 ablation group
    python run_ablation.py --groups component
    python run_ablation.py --groups k_sensitivity
    python run_ablation.py --groups ffe_sensitivity

    # 只跑一張圖快速驗證
    python run_ablation.py --image 0043 --n-runs 1

    # 自訂參數
    python run_ablation.py --image-size 1024 --range-size 4 --domain-size 8 --domain-stride 4

=============================================================================
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime
import csv

import fic_core as core
from encoders.feature_guided_pso import encode_feature_guided_pso
from encoders.feature_exhaustive import encode_feature_exhaustive


# =============================================================================
# Ablation configurations
# =============================================================================

def get_ablation_configs():
    """
    Returns dict of ablation_name -> (encoder_fn, kwargs_override).
    kwargs_override 會覆蓋 base FG-PSO config。
    """
    # Base FG-PSO config (same as configs/feature_guided_pso.yml)
    base = dict(
        top_k=40,
        feature_normalize=True,
        pop_size=20,
        max_iter=9999,  # will be capped by FFE budget
        w=0.9, c1=2.0, c2=2.0,
        v_max_ratio=0.2,
        early_stop_patience=9999,
        ls_strategies=('isometry', 'feature_neighbor'),
        ls_frequency=3,
        ls_top_percent=0.2,
        ls_at_end=True,
        feat_neighbors=4,
    )

    configs = {}

    # ---- Group: component ablation ----

    # Full FG-PSO (reference, same as main experiment)
    configs['fgpso_full'] = (
        encode_feature_guided_pso,
        {**base},
        'component'
    )

    # Feature-only exhaustive (no PSO, no LS)
    configs['feature_exhaustive'] = (
        encode_feature_exhaustive,
        dict(top_k=40, feature_normalize=True),
        'component'
    )

    # FG-PSO without any LS
    configs['fgpso_no_ls'] = (
        encode_feature_guided_pso,
        {**base, 'ls_strategies': (), 'ls_at_end': False},
        'component'
    )

    # FG-PSO with ISO-LS only (no Feature-Neighbor LS)
    configs['fgpso_iso_only'] = (
        encode_feature_guided_pso,
        {**base, 'ls_strategies': ('isometry',)},
        'component'
    )

    # ---- Group: K sensitivity ----

    for k in [10, 20, 40, 80, 160]:
        configs[f'fgpso_k{k}'] = (
            encode_feature_guided_pso,
            {**base, 'top_k': k},
            'k_sensitivity'
        )

    # ---- Group: FFE sensitivity ----
    # (These override the ffe_budget_per_block in the main loop)

    for b in [100, 200, 500, 1000]:
        configs[f'fgpso_b{b}'] = (
            encode_feature_guided_pso,
            {**base},
            'ffe_sensitivity'
        )
        # The FFE budget is injected in the main loop, not here

    return configs


# =============================================================================
# Image collection
# =============================================================================

def collect_images(image_dir, single_image=None):
    """Collect images to run."""
    if single_image:
        for ext in ['', '.png', '.jpg', '.bmp']:
            cand = os.path.join(image_dir, single_image + ext)
            if os.path.exists(cand):
                return [cand]
        if os.path.exists(single_image):
            return [single_image]
        print(f"Error: image '{single_image}' not found.")
        sys.exit(1)

    files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    if not files:
        print(f"No images found in {image_dir}/")
        sys.exit(1)
    return files


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FG-PSO Ablation Study Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--groups', nargs='+',
                        default=['component', 'k_sensitivity', 'ffe_sensitivity'],
                        choices=['component', 'k_sensitivity', 'ffe_sensitivity'],
                        help='Which ablation groups to run')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--range-size', type=int, default=4)
    parser.add_argument('--domain-size', type=int, default=8)
    parser.add_argument('--domain-stride', type=int, default=4)
    parser.add_argument('--ffe-budget', type=int, default=500,
                        help='Default FFE budget (overridden for ffe_sensitivity group)')
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='results_ablation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    all_configs = get_ablation_configs()

    # Filter by requested groups
    configs = {
        name: (fn, kw, grp)
        for name, (fn, kw, grp) in all_configs.items()
        if grp in args.groups
    }

    images = collect_images(args.image_dir, args.image)

    print(f"\n{'=' * 72}")
    print(f"  FG-PSO Ablation Study")
    print(f"{'=' * 72}")
    print(f"  Groups:        {args.groups}")
    print(f"  Ablations:     {list(configs.keys())}")
    print(f"  Images:        {[os.path.basename(p) for p in images]}")
    print(f"  Image size:    {args.image_size}x{args.image_size}")
    print(f"  FFE budget:    {args.ffe_budget} (default)")
    print(f"  Runs:          {args.n_runs}")
    print(f"  Output:        {args.output_dir}/")
    print(f"{'=' * 72}\n")

    all_results = []
    aggregated = defaultdict(list)

    for abl_name, (encoder_fn, kw_override, grp) in configs.items():

        # Determine FFE budget for this ablation
        if grp == 'ffe_sensitivity':
            # Extract budget from name: fgpso_b100 -> 100
            ffe_budget = int(abl_name.split('_b')[1])
        else:
            ffe_budget = args.ffe_budget

        for image_path in images:
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            for run_idx in range(args.n_runs):
                run_seed = args.seed + run_idx

                print(f"\n>>> Ablation={abl_name}  image={image_name}  "
                      f"run={run_idx+1}/{args.n_runs}  seed={run_seed}  "
                      f"ffe_budget={ffe_budget}")

                # Build kwargs
                kwargs = dict(kw_override)
                kwargs['seed'] = run_seed
                kwargs['ffe_budget_per_block'] = ffe_budget

                # For FG-PSO variants, override max_iter and early_stop
                if encoder_fn == encode_feature_guided_pso:
                    kwargs['max_iter'] = 9999
                    kwargs['early_stop_patience'] = 9999

                stats = core.run_pipeline(
                    encoder_fn, image_path,
                    method_name=abl_name,
                    output_dir=args.output_dir,
                    image_size=args.image_size,
                    range_size=args.range_size,
                    domain_size=args.domain_size,
                    domain_stride=args.domain_stride,
                    decode_iterations=20,
                    save_fic=False,
                    save_outputs=(run_idx == 0),
                    **kwargs,
                )
                stats['run_idx'] = run_idx
                stats['ablation'] = abl_name
                stats['ablation_group'] = grp
                stats['ffe_budget'] = ffe_budget
                all_results.append(stats)
                aggregated[(abl_name, image_name)].append(stats)

    # === Print summary ===
    print(f"\n\n{'=' * 120}")
    print(f"  ABLATION RESULTS (mean ± std over {args.n_runs} runs)")
    print(f"{'=' * 120}")
    print(f"  {'Ablation':<24} {'Image':<8} {'FFE':>6} {'Time(s)':>14} "
          f"{'PSNR(dB)':>16} {'AvgMSE':>18} {'Evals':>12}")
    print("-" * 120)

    for (abl_name, image_name), runs in sorted(aggregated.items()):
        t = np.array([r['encoding_time_sec'] for r in runs])
        p = np.array([r['psnr_actual'] for r in runs])
        m = np.array([r['mse_mean'] for r in runs])
        e = int(np.mean([r.get('n_evaluations', 0) for r in runs]))
        ffe = runs[0].get('ffe_budget', '?')

        if args.n_runs == 1:
            print(f"  {abl_name:<24} {image_name:<8} {ffe:>6} "
                  f"{t[0]:>14.2f} {p[0]:>16.2f} {m[0]:>18.4f} {e:>12,}")
        else:
            print(f"  {abl_name:<24} {image_name:<8} {ffe:>6} "
                  f"{t.mean():>7.2f}±{t.std():<5.2f} "
                  f"{p.mean():>8.2f}±{p.std():<6.2f} "
                  f"{m.mean():>9.4f}±{m.std():<7.4f} {e:>12,}")

    print("=" * 120)

    # === Save CSV ===
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"ablation_results_{timestamp}.csv")

    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    priority = ['ablation', 'ablation_group', 'image', 'run_idx', 'ffe_budget',
                'encoding_time_sec', 'psnr_actual', 'mse_mean', 'n_evaluations']
    keys = [k for k in priority if k in all_keys]
    keys += sorted(k for k in all_keys if k not in priority)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in keys}
            for k, v in row.items():
                if isinstance(v, (list, tuple, dict)):
                    row[k] = str(v)
            writer.writerow(row)

    print(f"\n  Saved CSV: {csv_path}")

    # === Save convergence data ===
    from convergence import save_convergence
    conv_groups = defaultdict(dict)
    for r in all_results:
        curve = r.get('convergence_curve')
        if curve:
            key = (r['image'], r.get('run_idx', 0))
            conv_groups[key][r['method']] = curve
    for (image_name, run_idx), curves in conv_groups.items():
        path = save_convergence(curves, args.output_dir, image_name, run_idx)
        print(f"  Saved convergence: {path}")


if __name__ == "__main__":
    main()
