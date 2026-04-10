"""
Segmentation Refinement Module
==============================
Post-processing operations to clean up raw segmentation masks.

Addresses common Graph Cut artifacts:
  - Noise:  small isolated foreground/background regions
  - Jagged edges:  non-smooth object boundaries
  - Intensity inconsistency:  fragmented regions that should be unified
"""

import cv2
import numpy as np


def refine_mask(mask, kernel_size=5, min_area_ratio=0.005):
    """
    Refine a binary segmentation mask using morphological operations
    and connected component filtering.

    Processing Steps
    ----------------
    1. Morphological Opening:  removes small isolated FG pixels (noise)
    2. Morphological Closing:  fills small holes inside the FG object
    3. Connected Component Filtering:  removes tiny disconnected regions
       whose area is below ``min_area_ratio * image_area``
    4. Gaussian Blur + Re-threshold:  smooths jagged boundaries for
       visually coherent edges

    Parameters
    ----------
    mask : ndarray, shape (H, W), values {0, 1}
        Raw binary segmentation mask from Graph Cut.
    kernel_size : int
        Size of the morphological structuring element (elliptical).
    min_area_ratio : float
        Minimum area of a connected component as a fraction of total
        image area.  Components smaller than this are removed.

    Returns
    -------
    mask_final : ndarray, shape (H, W), values {0, 1}, dtype uint8
        Refined binary mask with clean boundaries.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Step 1: Opening — erode then dilate (removes small noise patches)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 2: Closing — dilate then erode (fills small holes)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 3: Connected component filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    min_area = int(min_area_ratio * mask.shape[0] * mask.shape[1])

    cleaned = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    # Step 4: Gaussian blur for boundary smoothing
    blur_k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    mask_smoothed = cv2.GaussianBlur(cleaned, (blur_k, blur_k), 0)

    # Re-threshold to binary
    _, mask_final = cv2.threshold(mask_smoothed, 127, 255, cv2.THRESH_BINARY)

    return (mask_final // 255).astype(np.uint8)
