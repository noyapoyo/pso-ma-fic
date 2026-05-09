import argparse
import os
import sys
 
import fic_core as core
from encoders.full_search import encode_full_search
from encoders.pso import encode_pso
 
# encoder register
# encoder: name -> (function, default kwargs)
METHODS = {
    'full_search': (encode_full_search, {}),
    'pso': (encode_pso, {
        'pop_size': 40,
        'max_iter': 30,
        'w': 0.9,
        'c1': 2.0,
        'c2': 2.0,
    }),
}
 
 
def collect_images(image_dir, single_image=None):
    """collect images list"""
    if single_image:
        for ext in ['', '.png', '.jpg', '.bmp', '.tif', '.tiff', '.jpeg']:
            cand = os.path.join(image_dir, single_image + ext)
            if os.path.exists(cand):
                return [cand]
        if os.path.exists(single_image):
            return [single_image]
        print(f"Error: image '{single_image}' not found.")
        sys.exit(1)
 
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir}/ not found. Put test images there first.")
        sys.exit(1)
 
    files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    if not files:
        print(f"No images found in {image_dir}/")
        sys.exit(1)
    return files
 
 
def main():
    parser = argparse.ArgumentParser(description="FIC experiment runner")
    parser.add_argument('--methods', nargs='+', default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help='Execution algorithm (default: All)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image name or path (default: images/*)')
    # image arguments
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--image-size', type=int, default=256)

    # FIC arguments
    parser.add_argument('--range-size', type=int, default=4)
    parser.add_argument('--domain-size', type=int, default=8)

    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--decode-iter', type=int, default=20)
    args = parser.parse_args()
 
    images = collect_images(args.image_dir, args.image)
 
    print(f"\n{'*' * 70}")
    print(f"  FIC Experiment")
    print(f"  Methods:  {args.methods}")
    print(f"  Images:   {[os.path.basename(p) for p in images]}")
    print(f"  Output:   {args.output_dir}/")
    print(f"{'*' * 70}")
 
    all_stats = []
    for method_name in args.methods:
        encode_fn, default_kwargs = METHODS[method_name]
        for image_path in images:
            stats = core.run_pipeline(
                encode_fn, image_path, method_name=method_name,
                output_dir=args.output_dir,
                image_size=args.image_size,
                range_size=args.range_size,
                domain_size=args.domain_size,
                decode_iterations=args.decode_iter,
                **default_kwargs,
            )
            all_stats.append(stats)
 
    core.print_summary_table(all_stats)
 
 
if __name__ == "__main__":
    main()
