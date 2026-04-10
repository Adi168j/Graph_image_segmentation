"""
Graph Cut Image Segmentation Pipeline
======================================
Main entry point for the complete segmentation pipeline.

Pipeline Steps:
  1. Load input images with user-defined bounding boxes
  2. Naive baseline: Otsu's thresholding within the bounding box
  3. Graph Cut segmentation:
       a. Initialize mask from bounding box annotation
       b. Fit foreground / background GMMs (5 components each)
       c. Compute unary potentials (negative log-likelihoods)
       d. Build graph with unary + pairwise costs (8-neighborhood)
       e. Minimize energy via max-flow / min-cut (PyMaxflow)
       f. Update mask and iterate (5 iterations)
  4. Post-processing: morphological cleanup + boundary smoothing
  5. Visualization: comparison plots and overlays

Usage:
    python main.py

Output:
    Saves all results to output/ directory
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os
import time

from data_loader import load_images
from gmm import GMMModel
from graph_cut import GraphCutSegmenter
from refinement import refine_mask
from naive_seg import naive_segmentation


# ======================================================================
# Helper Functions
# ======================================================================

def init_mask_from_bbox(img_shape, bbox):
    """
    Initialize the segmentation mask from a user-provided bounding box.

    Convention (GrabCut-style):
        0 = Hard Background  (GC_BGD)      — outside bounding box
        1 = Hard Foreground  (GC_FGD)      — not used at initialization
        2 = Probable Background (GC_PR_BGD) — not used at initialization
        3 = Probable Foreground (GC_PR_FGD) — inside bounding box

    Parameters
    ----------
    img_shape : tuple (H, W, C)
    bbox : tuple (x, y, w, h) in pixels

    Returns
    -------
    mask : ndarray (H, W), dtype uint8
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = 3
    return mask


def create_overlay(img, mask, color=(0, 0, 255), alpha=0.4):
    """
    Create a semi-transparent color overlay on foreground pixels.
    """
    overlay = img.copy().astype(np.float64)
    fg = mask.astype(bool)
    overlay[fg] = overlay[fg] * (1 - alpha) + np.array(color, dtype=np.float64) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def extract_foreground(img, mask):
    """
    Extract foreground from image, setting background to white.
    """
    result = np.ones_like(img) * 255
    fg = mask.astype(bool)
    result[fg] = img[fg]
    return result


# ======================================================================
# Main Pipeline
# ======================================================================

def main():
    print("=" * 60)
    print("  Graph Cut Image Segmentation Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    MAX_DIM         = 500    # Resize images for tractable processing
    N_GMM_COMPONENTS = 5     # GMM components per model (GrabCut standard)
    N_ITERATIONS    = 5      # Number of iterative refinement steps
    GAMMA           = 50.0   # Smoothness weight for pairwise term

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading images...")
    images_info = load_images(data_dir="data", max_dim=MAX_DIM)
    print(f"  Loaded {len(images_info)} images")
    for info in images_info:
        h, w = info["image"].shape[:2]
        print(f"  - {info['name']}: {w}x{h}, bbox={info['bbox']}")

    # ------------------------------------------------------------------
    # Process Each Image
    # ------------------------------------------------------------------
    all_results = []

    for idx, info in enumerate(images_info):
        name = info["name"]
        img  = info["image"]
        bbox = info["bbox"]
        h, w = img.shape[:2]

        print(f"\n{'=' * 60}")
        print(f"  [{idx+1}/{len(images_info)}]  {name.upper()}")
        print(f"{'=' * 60}")

        # ==============================================================
        # Step 2: Naive Baseline
        # ==============================================================
        print("\n  [2/5] Naive segmentation (Otsu's thresholding)...")
        t0 = time.time()
        naive_mask = naive_segmentation(img, bbox)
        naive_time = time.time() - t0
        naive_overlay = create_overlay(img, naive_mask, color=(0, 255, 0))
        print(f"    Done in {naive_time:.2f}s | FG pixels: "
              f"{naive_mask.sum()}/{h*w} "
              f"({100*naive_mask.sum()/(h*w):.1f}%)")

        # ==============================================================
        # Step 3: Graph Cut Segmentation (Iterative)
        # ==============================================================
        print("\n  [3/5] Graph Cut segmentation...")
        t0 = time.time()

        # 3a. Initialize mask from bounding box
        mask = init_mask_from_bbox(img.shape, bbox)

        # 3b. Initialize segmenter — compute contrast-adaptive beta
        gc = GraphCutSegmenter(gamma=GAMMA)
        gc.compute_beta(img)
        print(f"    Beta (contrast param): {gc.beta:.6f}")

        # 3c. Initialize GMM model
        gmm = GMMModel(n_components=N_GMM_COMPONENTS)

        # Flatten image pixels for GMM input
        img_flat = img.reshape((-1, 3)).astype(np.float64)

        # 3d-f. Iterative optimization loop
        for it in range(N_ITERATIONS):
            # Current labels: FG (mask 1 or 3) → 1, BG (mask 0 or 2) → 0
            labels = np.where((mask == 1) | (mask == 3), 1, 0).flatten()

            # Fit GMMs to current labeling
            gmm.fit(img_flat, labels)

            if not gmm.is_fit:
                print(f"    Iteration {it+1}: GMM fitting failed, stopping.")
                break

            # Compute unary potentials from GMMs
            D_fg, D_bg = gmm.calculate_potentials(img_flat)

            # Build graph and compute min-cut
            new_labels = gc.build_graph_and_cut(img, D_fg, D_bg, mask)

            # Update mask: only modify probable pixels (2 or 3)
            mask = np.where(
                (mask == 2) | (mask == 3),
                np.where(new_labels == 1, 3, 2),
                mask
            )

            fg_count = np.sum((mask == 1) | (mask == 3))
            print(f"    Iteration {it+1}/{N_ITERATIONS}: "
                  f"FG = {fg_count}/{h*w} ({100*fg_count/(h*w):.1f}%)")

        gc_time = time.time() - t0
        print(f"    Graph Cut done in {gc_time:.2f}s")

        # Extract binary mask
        gc_mask = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
        gc_overlay = create_overlay(img, gc_mask, color=(0, 0, 255))

        # ==============================================================
        # Step 4: Post-Processing Refinement
        # ==============================================================
        print("\n  [4/5] Applying refinement (morphology + filtering)...")
        refined_mask = refine_mask(gc_mask)
        refined_overlay = create_overlay(img, refined_mask, color=(255, 100, 0))
        print(f"    Refined FG pixels: {refined_mask.sum()}/{h*w} "
              f"({100*refined_mask.sum()/(h*w):.1f}%)")

        # ==============================================================
        # Step 5: Save Results
        # ==============================================================
        print("\n  [5/5] Saving results...")

        # --- Individual outputs ---
        cv2.imwrite(os.path.join(out_dir, f"{name}_input.jpg"), img)
        cv2.imwrite(os.path.join(out_dir, f"{name}_naive_overlay.jpg"), naive_overlay)
        cv2.imwrite(os.path.join(out_dir, f"{name}_gc_overlay.jpg"), gc_overlay)
        cv2.imwrite(os.path.join(out_dir, f"{name}_refined_overlay.jpg"), refined_overlay)
        cv2.imwrite(os.path.join(out_dir, f"{name}_mask_naive.png"), naive_mask * 255)
        cv2.imwrite(os.path.join(out_dir, f"{name}_mask_gc.png"), gc_mask * 255)
        cv2.imwrite(os.path.join(out_dir, f"{name}_mask_refined.png"), refined_mask * 255)

        # Foreground extraction on white background
        fg_extract = extract_foreground(img, refined_mask)
        cv2.imwrite(os.path.join(out_dir, f"{name}_foreground.jpg"), fg_extract)

        # --- Comparison Plot (2 rows × 4 columns) ---
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(
            f"Graph Cut Segmentation — {name.capitalize()}",
            fontsize=18, fontweight='bold', y=0.98
        )

        # Row 1: Overlays
        img_rgb = img[:, :, ::-1].copy()
        x, y, bw, bh = bbox
        cv2.rectangle(img_rgb, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("Original + BBox", fontsize=13)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(naive_overlay[:, :, ::-1])
        axes[0, 1].set_title(f"Naive (Otsu)  [{naive_time:.2f}s]", fontsize=13)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gc_overlay[:, :, ::-1])
        axes[0, 2].set_title(f"Graph Cut (Raw)  [{gc_time:.2f}s]", fontsize=13)
        axes[0, 2].axis('off')

        axes[0, 3].imshow(refined_overlay[:, :, ::-1])
        axes[0, 3].set_title("Graph Cut (Refined)", fontsize=13)
        axes[0, 3].axis('off')

        # Row 2: Binary masks + foreground extraction
        axes[1, 0].imshow(img[:, :, ::-1])
        axes[1, 0].set_title("Original", fontsize=13)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(naive_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title("Naive Mask", fontsize=13)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(gc_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title("Graph Cut Mask", fontsize=13)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(refined_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 3].set_title("Refined Mask", fontsize=13)
        axes[1, 3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(
            os.path.join(out_dir, f"{name}_comparison.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

        # --- Foreground Extraction Plot ---
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
        fig2.suptitle(
            f"Foreground Extraction — {name.capitalize()}",
            fontsize=16, fontweight='bold'
        )
        axes2[0].imshow(img[:, :, ::-1])
        axes2[0].set_title("Original", fontsize=13)
        axes2[0].axis('off')

        axes2[1].imshow(fg_extract[:, :, ::-1])
        axes2[1].set_title("Extracted Foreground", fontsize=13)
        axes2[1].axis('off')

        plt.tight_layout()
        fig2.savefig(
            os.path.join(out_dir, f"{name}_extraction.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig2)

        all_results.append({
            "name": name,
            "naive_time": naive_time,
            "gc_time": gc_time,
            "naive_fg": int(naive_mask.sum()),
            "gc_fg": int(gc_mask.sum()),
            "refined_fg": int(refined_mask.sum()),
            "total_pixels": h * w,
        })

        print(f"  ✓ Saved all outputs for {name}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        tp = r['total_pixels']
        print(f"\n  {r['name'].capitalize()}:")
        print(f"    Naive:   {r['naive_fg']:>6d}/{tp} "
              f"({100*r['naive_fg']/tp:.1f}%) — {r['naive_time']:.2f}s")
        print(f"    GC:      {r['gc_fg']:>6d}/{tp} "
              f"({100*r['gc_fg']/tp:.1f}%) — {r['gc_time']:.2f}s")
        print(f"    Refined: {r['refined_fg']:>6d}/{tp} "
              f"({100*r['refined_fg']/tp:.1f}%)")

    print(f"\n  All results saved to {out_dir}/")
    print("  Done!")


if __name__ == "__main__":
    main()
