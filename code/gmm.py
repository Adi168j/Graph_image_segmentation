"""
Gaussian Mixture Model (GMM) Module
====================================
Models pixel color distributions for foreground and background using GMMs.

Two separate GMMs (each with K components) are fit:
  - GMM_fg : models the color distribution of foreground pixels
  - GMM_bg : models the color distribution of background pixels

The negative log-likelihoods from these models serve as the unary (data)
term in the energy function:
  D_fg(p) = -log P(I_p | GMM_fg)   — cost of labeling pixel p as foreground
  D_bg(p) = -log P(I_p | GMM_bg)   — cost of labeling pixel p as background

References:
  - Rother et al., "GrabCut", SIGGRAPH 2004 (uses 5-component full-cov GMMs)
"""

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMModel:
    """
    Foreground / Background color model using Gaussian Mixture Models.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components per model.  GrabCut uses 5.
    """

    def __init__(self, n_components=5):
        self.n_components = n_components
        self.gmm_fg = None
        self.gmm_bg = None
        self.is_fit = False

    def fit(self, X, labels):
        """
        Fit foreground and background GMMs to the current pixel assignments.

        Parameters
        ----------
        X : ndarray, shape (N, 3)
            Pixel colors (BGR) for all N pixels in the image.
        labels : ndarray, shape (N,)
            Current label assignment: 1 = foreground, 0 = background.
        """
        fg_pixels = X[labels == 1]
        bg_pixels = X[labels == 0]

        # Need at least n_components samples to fit
        fg_ok = len(fg_pixels) >= self.n_components
        bg_ok = len(bg_pixels) >= self.n_components

        if fg_ok:
            self.gmm_fg = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                random_state=42,
                max_iter=200,
            )
            self.gmm_fg.fit(fg_pixels)

        if bg_ok:
            self.gmm_bg = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                random_state=42,
                max_iter=200,
            )
            self.gmm_bg.fit(bg_pixels)

        self.is_fit = fg_ok and bg_ok

    def calculate_potentials(self, X):
        """
        Compute unary potentials (negative log-likelihoods) for every pixel.

        The cost of assigning pixel p to foreground is proportional to how
        unlikely its color is under the foreground model, and vice versa.

        Parameters
        ----------
        X : ndarray, shape (N, 3)

        Returns
        -------
        D_fg : ndarray (N,) — cost to label as FG = -log P(pixel | FG_GMM)
        D_bg : ndarray (N,) — cost to label as BG = -log P(pixel | BG_GMM)
        """
        if not self.is_fit:
            raise ValueError("GMMs must be fit before calculating potentials.")

        # sklearn score_samples returns log P(x) (log-probability)
        log_prob_fg = self.gmm_fg.score_samples(X)
        log_prob_bg = self.gmm_bg.score_samples(X)

        D_fg = -log_prob_fg
        D_bg = -log_prob_bg

        return D_fg, D_bg
