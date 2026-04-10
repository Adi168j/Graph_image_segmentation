"""
Graph Cut Segmentation Module
=============================
Implements the core graph-based energy minimization for image segmentation.

The image is represented as a graph where:
  - Each pixel is a node
  - Terminal edges (t-links) connect nodes to Source (FG) and Sink (BG)
  - Neighborhood edges (n-links) connect adjacent pixels (8-connected)

Energy Function:
    E(L) = sum_p E_data(L_p)  +  sum_{(p,q) in N} E_smooth(L_p, L_q)

where:
  - E_data : unary cost from GMM negative log-likelihoods
  - E_smooth : pairwise cost penalizing label discontinuities at weak edges,
               using the Boykov-Jolly formulation with contrast-adaptive beta

References:
  - Boykov & Jolly, "Interactive Graph Cuts for Optimal Boundary & Region
    Segmentation of Objects in N-D Images", ICCV 2001
  - Rother et al., "GrabCut — Interactive Foreground Extraction using
    Iterated Graph Cuts", SIGGRAPH 2004
"""

import numpy as np
import maxflow


class GraphCutSegmenter:
    """
    Graph Cut segmentation using 8-neighborhood connectivity.

    Parameters
    ----------
    gamma : float
        Weight for the smoothness (pairwise) term.  Higher gamma = stronger
        preference for spatial coherence.  Typical range: 10–100.
    """

    def __init__(self, gamma=50.0):
        self.gamma = gamma
        self.beta = 0.0

    # ------------------------------------------------------------------
    # Beta Computation
    # ------------------------------------------------------------------
    def compute_beta(self, img):
        """
        Compute the contrast parameter beta from the image.

        Beta controls the sensitivity of pairwise costs to color differences.
        It is set to the inverse of twice the expected squared color difference
        across all neighboring pixel pairs (8-connected):

            beta = 1 / (2 * <||I_p - I_q||^2>)

        This ensures that the smoothness term adapts to image contrast:
          - High-contrast images → smaller beta → edges are more permeable
          - Low-contrast images  → larger beta  → edges are less permeable

        Parameters
        ----------
        img : ndarray, shape (H, W, 3)
            Input image in BGR format.
        """
        img_f = img.astype(np.float64)

        # Compute squared differences for 4 unique neighbor directions
        # (right, down, down-right, down-left) to avoid double-counting
        diff_right = img_f[:, 1:]  - img_f[:, :-1]
        diff_down  = img_f[1:, :]  - img_f[:-1, :]
        diff_dr    = img_f[1:, 1:] - img_f[:-1, :-1]    # diagonal ↘
        diff_dl    = img_f[1:, :-1] - img_f[:-1, 1:]     # diagonal ↙

        sq_right = np.sum(diff_right ** 2, axis=-1)
        sq_down  = np.sum(diff_down  ** 2, axis=-1)
        sq_dr    = np.sum(diff_dr    ** 2, axis=-1)
        sq_dl    = np.sum(diff_dl    ** 2, axis=-1)

        total_sq = sq_right.sum() + sq_down.sum() + sq_dr.sum() + sq_dl.sum()
        count    = sq_right.size  + sq_down.size  + sq_dr.size  + sq_dl.size

        if total_sq == 0:
            self.beta = 0.0
        else:
            self.beta = 1.0 / (2.0 * total_sq / count)

    # ------------------------------------------------------------------
    # Pairwise (Smoothness) Costs
    # ------------------------------------------------------------------
    def compute_pairwise(self, img):
        """
        Compute pairwise (smoothness) edge capacities for 8-neighborhood.

        The smoothness cost between adjacent pixels p, q is:

            V(p, q) = gamma * exp(-beta * ||I_p - I_q||^2) / dist(p,q)

        Properties:
          - Similar neighbors  → high cost → discourages cutting between them
          - Dissimilar neighbors → low cost  → encourages cutting at boundaries
          - Diagonal edges are divided by sqrt(2) for Euclidean distance

        Parameters
        ----------
        img : ndarray, shape (H, W, 3)

        Returns
        -------
        smooth_right, smooth_down, smooth_dr, smooth_dl : ndarray (H, W)
            Edge capacities for each direction, zero-padded to image dims.
        """
        img_f = img.astype(np.float64)

        # Squared color differences
        sq_right = np.sum((img_f[:, 1:]  - img_f[:, :-1])  ** 2, axis=-1)
        sq_down  = np.sum((img_f[1:, :]  - img_f[:-1, :])  ** 2, axis=-1)
        sq_dr    = np.sum((img_f[1:, 1:] - img_f[:-1, :-1]) ** 2, axis=-1)
        sq_dl    = np.sum((img_f[1:, :-1] - img_f[:-1, 1:]) ** 2, axis=-1)

        # Smoothness penalties (Boykov-Jolly formulation)
        smooth_right = self.gamma * np.exp(-self.beta * sq_right)
        smooth_down  = self.gamma * np.exp(-self.beta * sq_down)
        smooth_dr    = (self.gamma / np.sqrt(2.0)) * np.exp(-self.beta * sq_dr)
        smooth_dl    = (self.gamma / np.sqrt(2.0)) * np.exp(-self.beta * sq_dl)

        # Pad to full image dimensions
        smooth_right = np.pad(smooth_right, ((0, 0), (0, 1)), mode='constant')
        smooth_down  = np.pad(smooth_down,  ((0, 1), (0, 0)), mode='constant')
        smooth_dr    = np.pad(smooth_dr,    ((0, 1), (0, 1)), mode='constant')
        smooth_dl    = np.pad(smooth_dl,    ((0, 1), (1, 0)), mode='constant')

        return smooth_right, smooth_down, smooth_dr, smooth_dl

    # ------------------------------------------------------------------
    # Graph Construction + Min-Cut
    # ------------------------------------------------------------------
    def build_graph_and_cut(self, img, D_fg, D_bg, mask):
        """
        Construct the graph and compute the minimum cut.

        Graph Structure
        ---------------
        - Source node (S) represents Foreground
        - Sink   node (T) represents Background
        - Each pixel node p has:
            * t-link to S with capacity = D_bg[p]  (cost of labeling BG)
            * t-link to T with capacity = D_fg[p]  (cost of labeling FG)
            * n-links to 8 neighbors with smoothness capacities

        Hard Constraints (from user annotations)
        -----------------------------------------
        - mask == 0 (Hard BG):  S→p = 0,         p→T = LARGE
        - mask == 1 (Hard FG):  S→p = LARGE,     p→T = 0
        - mask == 2 (Prob BG):  uses GMM potentials
        - mask == 3 (Prob FG):  uses GMM potentials

        The Max-Flow / Min-Cut theorem guarantees that the maximum flow
        equals the minimum cut, partitioning nodes into
        S-reachable (FG) and T-reachable (BG).

        Parameters
        ----------
        img  : ndarray (H, W, 3) — input image (BGR)
        D_fg : ndarray (H*W,)    — unary cost for FG:  -log P(px | FG_GMM)
        D_bg : ndarray (H*W,)    — unary cost for BG:  -log P(px | BG_GMM)
        mask : ndarray (H, W)    — annotation mask

        Returns
        -------
        labels : ndarray (H, W), dtype uint8 — 1=FG, 0=BG
        """
        h, w = img.shape[:2]
        num_nodes = h * w

        # --- 1. Create graph ---
        g = maxflow.Graph[float](num_nodes, num_nodes * 8)
        nodes = g.add_nodes(num_nodes)
        node_ids = np.arange(num_nodes).reshape((h, w))

        # --- 2. Terminal edges (t-links / unary costs) ---
        D_bg_2d = D_bg.reshape((h, w))
        D_fg_2d = D_fg.reshape((h, w))
        LARGE_VAL = 1e9

        # Source weights: capacity of edge  S → pixel
        #   - If pixel is Hard BG  → 0          (no reason to connect to S)
        #   - If pixel is Hard FG  → LARGE_VAL  (must stay connected to S)
        #   - Otherwise            → D_bg       (cost of choosing BG label)
        source_w = np.where(
            mask == 0, 0,
            np.where(mask == 1, LARGE_VAL, D_bg_2d)
        )

        # Sink weights: capacity of edge  pixel → T
        #   - If pixel is Hard BG  → LARGE_VAL  (must stay connected to T)
        #   - If pixel is Hard FG  → 0          (no reason to connect to T)
        #   - Otherwise            → D_fg       (cost of choosing FG label)
        sink_w = np.where(
            mask == 0, LARGE_VAL,
            np.where(mask == 1, 0, D_fg_2d)
        )

        g.add_grid_tedges(node_ids, source_w, sink_w)

        # --- 3. Neighborhood edges (n-links / pairwise costs) ---
        smooth_r, smooth_d, smooth_dr, smooth_dl = self.compute_pairwise(img)

        # Structure matrices: center pixel is [1,1]; the 1 marks neighbor
        struct_r  = np.array([[0,0,0],[0,0,1],[0,0,0]])   # →
        struct_d  = np.array([[0,0,0],[0,0,0],[0,1,0]])   # ↓
        struct_dr = np.array([[0,0,0],[0,0,0],[0,0,1]])   # ↘
        struct_dl = np.array([[0,0,0],[0,0,0],[1,0,0]])   # ↙

        g.add_grid_edges(node_ids, smooth_r,  struct_r,  symmetric=True)
        g.add_grid_edges(node_ids, smooth_d,  struct_d,  symmetric=True)
        g.add_grid_edges(node_ids, smooth_dr, struct_dr, symmetric=True)
        g.add_grid_edges(node_ids, smooth_dl, struct_dl, symmetric=True)

        # --- 4. Compute Max-Flow (= Min-Cut) ---
        flow = g.maxflow()

        # --- 5. Extract labels (vectorized) ---
        # get_grid_segments: True → Sink (BG), False → Source (FG)
        segments = g.get_grid_segments(node_ids)
        labels = np.where(segments, 0, 1).astype(np.uint8)

        return labels
