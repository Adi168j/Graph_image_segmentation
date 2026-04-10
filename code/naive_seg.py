"""
Naive Segmentation Baseline
============================
Provides a simple threshold-based segmentation for comparison against
the Graph Cut method.

Uses Otsu's automatic thresholding on the grayscale ROI defined by the
bounding box.  This serves as a lower-bound baseline:  it considers only
pixel intensity and ignores spatial context and color distributions.
"""

import cv2
import numpy as np


def naive_segmentation(img, bbox):
    """
    Segment the image using Otsu's thresholding within the bounding box.

    Limitations (compared to Graph Cut):
      - No spatial coherence — each pixel is classified independently
      - Assumes bimodal intensity distribution — fails on complex textures
      - No iterative refinement — one-shot classification
      - No color modeling — uses only grayscale intensity

    Parameters
    ----------
    img : ndarray, shape (H, W, 3), BGR
        Input image.
    bbox : tuple (x, y, w, h)
        Bounding box defining the region of interest.

    Returns
    -------
    mask : ndarray, shape (H, W), values {0, 1}, dtype uint8
        Binary segmentation mask.
    """
    x, y, w, h = bbox
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    roi = gray[y:y+h, x:x+w]

    # Otsu's method finds the optimal threshold that minimizes
    # intra-class variance, assuming a bimodal histogram
    _, thresh = cv2.threshold(roi, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heuristic: assume the object is near the center of the bbox
    # If the center pixel is labeled 0, invert so FG = 1
    center_y, center_x = h // 2, w // 2
    if thresh[center_y, center_x] == 0:
        thresh = 1 - thresh

    mask[y:y+h, x:x+w] = thresh
    return mask
