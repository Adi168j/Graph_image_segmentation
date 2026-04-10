"""
Data Loading Module for Graph Cut Segmentation Pipeline
=======================================================
Loads user-provided images and defines foreground bounding boxes
to guide the initial segmentation mask.

Images:
  1. Chameleon  — natural scene, green chameleon on a branch
  2. Spider-Man — comic art, character swinging between buildings
  3. Venom      — dark art, character face against dark background
"""
import os
import cv2
import numpy as np


def load_images(data_dir="data", max_dim=500):
    """
    Load the three user-provided images and configure bounding boxes.

    Each bounding box (x, y, w, h) marks the approximate foreground region.
    Pixels outside the box are treated as Hard Background; pixels inside
    are initialized as Probable Foreground for the Graph Cut algorithm.

    Parameters
    ----------
    data_dir : str
        Directory containing the input images.
    max_dim : int
        Maximum dimension (width or height) to resize images to.
        Larger images are scaled down for tractable graph construction.

    Returns
    -------
    list of dict
        Each dict has keys: name, file, image (BGR ndarray), bbox (x,y,w,h)
    """
    image_configs = [
        {
            "name": "chameleon",
            "file": "chameleon.jpg",
            # BBox covers the chameleon body (left-center of image)
            # Specified as fractions of (x_start, y_start, width, height)
            "bbox_frac": (0.02, 0.06, 0.72, 0.88),
        },
        {
            "name": "spiderman",
            "file": "spiderman.jpg",
            # BBox covers Spider-Man figure (central majority of frame)
            "bbox_frac": (0.08, 0.02, 0.85, 0.95),
        },
        {
            "name": "venom",
            "file": "venom.jpg",
            # BBox covers Venom's face and upper body (center)
            "bbox_frac": (0.12, 0.03, 0.70, 0.94),
        },
    ]

    images_info = []
    for cfg in image_configs:
        path = os.path.join(data_dir, cfg["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Image not found: {path}\n"
                f"Please place '{cfg['file']}' in the '{data_dir}/' directory."
            )

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        # Resize if too large (graph cut is O(N) in pixels for graph construction)
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Convert fractional bbox to pixel coordinates
        fx, fy, fw, fh = cfg["bbox_frac"]
        bbox = (int(fx * w), int(fy * h), int(fw * w), int(fh * h))

        images_info.append({
            "name": cfg["name"],
            "file": path,
            "image": img,
            "bbox": bbox,
        })

    return images_info
