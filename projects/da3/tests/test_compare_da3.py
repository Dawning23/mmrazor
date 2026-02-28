"""
Comparison test: mmrazor DA3 (FP32) vs saved reference predictions.

Loads the mmrazor DA3 model via DA3Inferencer, runs FP32 inference on SOH
images, and compares the results against pre-saved reference predictions
from the original DA3 model (/prediction_fp32.npz).

Usage:
    python projects/da3/tests/test_compare_da3.py
    python projects/da3/tests/test_compare_da3.py --visualize
"""

import argparse
import glob
import sys
import os
import time
import torch
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# ── Paths ──────────────────────────────────────────────────────────────────────
SAVED_WEIGHTS_PATH = "/home/drobotics/bohaozhang/da3_large_weights.pth"
DA3_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Depth-Anything-3')
SOH_PATH = os.path.join(DA3_ROOT, 'assets', 'examples', 'SOH')
REFERENCE_NPZ = "/home/drobotics/bohaozhang/Depth-Anything-3/src/prediction_fp32.npz"
VIS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare_numpy(name, a, b, rtol=1e-5, atol=1e-5):
    """Compare two numpy arrays and print results."""
    if a is None and b is None:
        print(f"  {name:25s} [both None] — skipped")
        return True
    if a is None or b is None:
        print(f"  {name:25s} ✗ FAIL  one is None")
        return False
    if a.shape != b.shape:
        print(f"  {name:25s} ✗ FAIL  shape mismatch: {a.shape} vs {b.shape}")
        return False

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    max_diff = np.abs(a_f - b_f).max()
    mean_diff = np.abs(a_f - b_f).mean()
    rel_max = (np.abs(a_f - b_f) / (np.abs(b_f) + 1e-8)).max()
    is_close = np.allclose(a_f, b_f, rtol=rtol, atol=atol)
    status = "✓ PASS" if is_close else "✗ FAIL"
    print(f"  {name:25s} {status}  max_diff={max_diff:.2e}  "
          f"mean_diff={mean_diff:.2e}  rel_max={rel_max:.2e}  shape={list(a.shape)}")
    return is_close


# ── Visualization ──────────────────────────────────────────────────────────────

def depth_to_colormap(depth, cmap=cv2.COLORMAP_INFERNO):
    d = depth.astype(np.float64)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-8:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    return cv2.applyColorMap(d_norm, cmap)


def visualize_comparison(ref_data, mmr_pred):
    """Save side-by-side depth comparison images."""
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    ref_depth = ref_data['depth']
    mmr_depth = mmr_pred.depth

    n_images = ref_depth.shape[0]
    for i in range(n_images):
        ref_cmap = depth_to_colormap(ref_depth[i])
        mmr_cmap = depth_to_colormap(mmr_depth[i])

        # Difference map
        diff = np.abs(ref_depth[i].astype(np.float64) - mmr_depth[i].astype(np.float64))
        diff_cmap = depth_to_colormap(diff, cmap=cv2.COLORMAP_HOT)

        # RGB panel from reference processed images
        rgb_panel = ref_data['processed_images'][i]
        rgb_bgr = cv2.cvtColor(rgb_panel, cv2.COLOR_RGB2BGR)

        canvas = np.concatenate([rgb_bgr, ref_cmap, mmr_cmap, diff_cmap], axis=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        pw = rgb_panel.shape[1]
        cv2.putText(canvas, "RGB", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Reference (Original FP32)", (pw + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "mmrazor FP32", (pw * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Difference", (pw * 3 + 10, 30), font, 0.7, (255, 255, 255), 2)

        out_path = os.path.join(VIS_OUTPUT_DIR, f"fp32_compare_{i:03d}.png")
        cv2.imwrite(out_path, canvas)
        print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare mmrazor DA3 (FP32) vs saved reference predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true",
                        help="Save side-by-side depth comparison images")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for comparison")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for comparison")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("DA3 FP32 Comparison: mmrazor vs Reference (Original)")
    print("=" * 60)

    # ── Step 1: Load reference predictions ──
    print(f"\n[1/3] Loading reference predictions from:")
    print(f"  {REFERENCE_NPZ}")
    assert os.path.exists(REFERENCE_NPZ), f"Reference file not found: {REFERENCE_NPZ}"
    ref = np.load(REFERENCE_NPZ)
    print(f"  Keys: {list(ref.keys())}")
    for k in ref.keys():
        print(f"    {k:20s} shape={list(ref[k].shape)}  dtype={ref[k].dtype}")

    # ── Step 2: Build mmrazor model from config ──
    print("\n[2/3] Building mmrazor DA3 from config...")
    from projects.da3.configs.da3_large import da3_large_cfg
    from projects.da3.models.inferencer import DA3Inferencer

    t0 = time.time()
    mmr_inferencer = DA3Inferencer.from_config(da3_large_cfg)
    t1 = time.time()
    print(f"  Built in {t1 - t0:.2f}s")
    print(f"  Device: {da3_large_cfg['device']}")

    # ── Step 3: Run mmrazor inference on same images ──
    print("\n[3/3] Running mmrazor FP32 inference...")
    images = sorted(glob.glob(os.path.join(SOH_PATH, "*.png")))
    assert len(images) > 0, f"No images found in {SOH_PATH}"
    print(f"  Images: {[os.path.basename(p) for p in images]}")

    t0 = time.time()
    mmr_pred = mmr_inferencer.inference(images)
    t1 = time.time()
    print(f"  Inference time: {t1 - t0:.2f}s")

    # ── Output shapes ──
    print(f"\n  mmrazor output shapes:")
    print(f"    depth:      {mmr_pred.depth.shape}")
    if mmr_pred.conf is not None:
        print(f"    conf:       {mmr_pred.conf.shape}")
    if mmr_pred.extrinsics is not None:
        print(f"    extrinsics: {mmr_pred.extrinsics.shape}")
    if mmr_pred.intrinsics is not None:
        print(f"    intrinsics: {mmr_pred.intrinsics.shape}")

    # ── Comparison ──
    print(f"\n{'=' * 60}")
    print(f"Output Comparison: mmrazor FP32 vs Reference FP32")
    print(f"  rtol={args.rtol:.0e}, atol={args.atol:.0e}")
    print(f"{'=' * 60}")
    all_close = True
    all_close &= compare_numpy("depth", mmr_pred.depth, ref['depth'],
                               rtol=args.rtol, atol=args.atol)
    all_close &= compare_numpy("conf", mmr_pred.conf, ref['conf'],
                               rtol=args.rtol, atol=args.atol)
    all_close &= compare_numpy("extrinsics", mmr_pred.extrinsics, ref['extrinsics'],
                               rtol=args.rtol, atol=args.atol)
    all_close &= compare_numpy("intrinsics", mmr_pred.intrinsics, ref['intrinsics'],
                               rtol=args.rtol, atol=args.atol)

    # Also compare processed images
    if mmr_pred.processed_images is not None and 'processed_images' in ref:
        all_close &= compare_numpy("processed_images",
                                   mmr_pred.processed_images, ref['processed_images'],
                                   rtol=0, atol=1)  # uint8, allow ±1

    print()
    if all_close:
        print("✓ ALL OUTPUTS MATCH within tolerance!")
    else:
        print("✗ SOME OUTPUTS DO NOT MATCH")

    # ── Visualization ──
    if args.visualize:
        print(f"\n{'=' * 60}")
        print("Generating comparison visualizations...")
        print(f"{'=' * 60}")
        visualize_comparison(ref, mmr_pred)

    return 0 if all_close else 1


if __name__ == "__main__":
    sys.exit(main())
