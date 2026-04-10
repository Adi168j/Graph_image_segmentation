# Image Segmentation using Graph Cut

This report details the implementation, methodology, and evaluation of an Image Segmentation pipeline utilizing the Graph Cut technique to separate foreground and background regions from an image. The pipeline closely replicates the GrabCut algorithm using Python, scikit-learn (for color modeling), and PyMaxflow (for min-cut/max-flow optimization).

## Input Images

Three diverse images are used to evaluate the segmentation pipeline:

| # | Image | Type | Challenge |
|---|-------|------|-----------|
| 1 | **Chameleon** | Natural scene | Green foreground against green blurred background — tests color similarity handling |
| 2 | **Spider-Man** | Comic art | Multi-colored foreground (red/blue/black) against complex building background — tests multi-modal color distributions |
| 3 | **Venom** | Dark art | Dark foreground against dark background — tests low-contrast segmentation |

Each image is annotated with a **bounding box** that marks the approximate foreground region. All pixels outside the bounding box are treated as definite background; pixels inside are treated as probable foreground and refined through iterative optimization.

## Methodology

### 1. Energy Formulation

The segmentation is formulated as an energy minimization problem, where we assign a binary label $L_p \in \{0, 1\}$ (Background, Foreground) to each pixel $p$. The energy function to minimize is:

$$E(L) = \underbrace{\sum_{p} E_{data}(L_p)}_{\text{Unary Term}} + \underbrace{\sum_{(p,q) \in \mathcal{N}} E_{smooth}(L_p, L_q)}_{\text{Pairwise Term}}$$

- **$E_{data}$ (Unary Cost)**: Represents the penalty of assigning a pixel to the foreground or background, derived from the pixel's likelihood under learned color models.
- **$E_{smooth}$ (Pairwise Cost)**: Represents the penalty of assigning different labels to adjacent pixels, encouraging spatial coherence and smooth boundaries.

### 2. Foreground-Background Modeling (GMM)

Pixel likelihoods are modeled using **Gaussian Mixture Models (GMMs)**. Two separate GMMs (each with K=5 components, full covariance) are trained: one for foreground pixels and one for background pixels.

- The initial mask is generated using a user-provided bounding box. The area outside the box is initialized as **Hard Background**, while the interior is treated as **Probable Foreground**.
- The negative log-likelihoods from the GMMs serve as the unary (data) cost:
  - $D_{fg}(p) = -\log P(I_p \mid GMM_{fg})$ — cost if pixel $p$ is labeled as Foreground
  - $D_{bg}(p) = -\log P(I_p \mid GMM_{bg})$ — cost if pixel $p$ is labeled as Background

### 3. Graph Construction

To apply the min-cut framework, we construct a directed graph where:

- **Nodes**: Each pixel is represented as a node.
- **Terminal Nodes**: Two special nodes — **Source (S)** representing Foreground and **Sink (T)** representing Background.

**Terminal Edges (t-links)**:
- Edge capacity from Source to node $p$: $D_{bg}(p)$ — cost of labeling $p$ as Background
- Edge capacity from node $p$ to Sink: $D_{fg}(p)$ — cost of labeling $p$ as Foreground
- Hard constraints enforce annotations via very large capacities ($10^9$)

**Neighborhood Edges (n-links)**:
Adjacent pixels are connected using 8-neighborhood connectivity with capacities:

$$V(p, q) = \gamma \cdot \exp\left(-\beta \|I_p - I_q\|^2\right) / \text{dist}(p, q)$$

where:
- $\gamma = 50$ is the smoothness weight
- $\beta = \frac{1}{2 \langle \|I_p - I_q\|^2 \rangle}$ adapts to image contrast
- Diagonal edges are divided by $\sqrt{2}$ for Euclidean distance normalization

This formulation gives **high cost** between similar neighbors (discouraging cuts) and **low cost** between dissimilar neighbors (encouraging cuts at object boundaries).

### 4. Min-Cut / Max-Flow

The optimal segmentation is discovered using the **PyMaxflow** library by computing the maximum flow from Source to Sink. By the Max-Flow/Min-Cut theorem, the edges that saturate define the **Minimum Cut**, partitioning every pixel into either:
- **Source tree** (Foreground) — connected to Source after the cut
- **Sink tree** (Background) — connected to Sink after the cut

### 5. Iterative Optimization

The entire process is nested in an iterative loop (5 iterations):

1. **Assign labels** from the current mask (probable pixels only)
2. **Re-fit** foreground and background GMMs on current assignments
3. **Compute** new unary potentials using the fitted GMMs
4. **Apply Min-Cut** to find optimal labeling
5. **Update mask** — only modify probable pixels (hard constraints are preserved)
6. **Repeat** until convergence (typically 3–5 iterations)

### 6. Artifact Mitigation & Refinement

The raw segmentation output often contains jagged edges or small isolated noise fragments. To produce a clean final mask:

1. **Morphological Opening** (elliptical kernel, 2 iterations): Removes thin protrusions and isolated noise pixels.
2. **Morphological Closing** (elliptical kernel, 2 iterations): Seals small holes inside the foreground object.
3. **Connected Component Filtering**: Removes disconnected foreground regions smaller than 0.5% of image area.
4. **Gaussian Blur + Re-threshold**: Smooths the boundary for visually coherent edges.

## Comparison: Naive vs Graph Cut

### Naive Segmentation (Otsu's Thresholding)
- Applies Otsu's automatic thresholding on the grayscale ROI within the bounding box
- **Limitations**:
  - No spatial coherence — each pixel classified independently based on intensity
  - Assumes bimodal intensity distribution — fails on complex textures
  - Uses only grayscale — ignores rich color information
  - One-shot classification — no iterative refinement

### Graph Cut Segmentation
- **Advantages over Naive**:
  - Spatial coherence through pairwise smoothness terms
  - Full color modeling (BGR) via 5-component GMMs per class
  - Iterative optimization that progressively refines the boundary
  - Contrast-adaptive edge weights that respect true object boundaries
  - Hard constraints from user annotations provide reliable seeds

## Per-Image Observations

### Chameleon (Natural Scene)
- **Challenge**: The green chameleon is set against a green blurred background, creating significant color overlap between foreground and background.
- **Naive**: Struggles severely because the intensity distributions of chameleon and foliage overlap heavily.
- **Graph Cut**: Successfully leverages the texture differences (sharp chameleon vs. blurred background) through pairwise terms, and the GMM captures subtle color distribution differences in the full BGR space.
- **Refined**: Clean boundaries around the chameleon body with minimal background leakage.

### Spider-Man (Comic Art)
- **Challenge**: Multi-colored foreground (red suit, blue accents, black spider emblem, white webs) against a complex urban background with multiple building colors.
- **Naive**: Fails badly as the foreground contains multiple distinct color modes, violating the bimodal assumption.
- **Graph Cut**: The 5-component GMM handles the multi-modal foreground distribution well, learning separate components for red, blue, black, and white regions.
- **Refined**: Sharp boundaries around the character with buildings correctly classified as background.

### Venom (Dark Art)
- **Challenge**: Very dark foreground (black symbiote) against a dark background, creating extremely low contrast at object boundaries.
- **Naive**: Essentially random labeling in extremely low-contrast regions.
- **Graph Cut**: Relies heavily on the pairwise term to enforce spatial coherence, as color differences are minimal. The white eyes and teeth provide strong anchor points for the foreground model.
- **Refined**: Handles the subtle boundary between Venom and the dark background.

## Output Files

For each image, the pipeline produces:

| File | Description |
|------|-------------|
| `*_input.jpg` | Resized input image |
| `*_naive_overlay.jpg` | Naive segmentation overlay (green) |
| `*_gc_overlay.jpg` | Graph Cut overlay (blue) |
| `*_refined_overlay.jpg` | Refined Graph Cut overlay (orange) |
| `*_mask_naive.png` | Binary mask — naive method |
| `*_mask_gc.png` | Binary mask — raw Graph Cut |
| `*_mask_refined.png` | Binary mask — refined Graph Cut |
| `*_foreground.jpg` | Extracted foreground on white background |
| `*_comparison.png` | Side-by-side comparison plot |
| `*_extraction.png` | Foreground extraction visualization |

## Conclusion

The Graph Cut segmentation pipeline significantly outperforms the naive Otsu baseline across all three test images. The combination of GMM color modeling, contrast-adaptive pairwise costs, and iterative optimization produces accurate segmentation even in challenging scenarios (color similarity, multi-modal distributions, and low contrast). The post-processing refinement further improves the visual quality by removing noise and smoothing boundaries.
