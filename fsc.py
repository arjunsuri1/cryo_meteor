"""Fourier shell correlation: the cryo-EM-native objective replacing negentropy."""
from __future__ import annotations

import numpy as np


def fsc_curve(a: np.ndarray, b: np.ndarray, nbins: int = 50) -> np.ndarray:  
    A, B = np.fft.fftn(a), np.fft.fftn(b)
    n = a.shape[0]
    f = np.fft.fftfreq(n) * n      
    kx, ky, kz = np.meshgrid(f, f, f, indexing="ij")
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    edges = np.linspace(0, r.max(), nbins + 1)
    idx = np.digitize(r, edges)
    out = np.full(nbins, np.nan)
    for i in range(1, nbins + 1):
        m = idx == i
        if not m.any():
            continue
        num = np.sum(A[m] * np.conj(B[m])).real
        den = np.sqrt(np.sum(np.abs(A[m]) ** 2) * np.sum(np.abs(B[m]) ** 2))
        out[i - 1] = num / den if den else np.nan
    return out


def fsc_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(fsc_curve(a, b)))


  """FSC between two equal-shape real-space maps, as `nbins` shell correlations.

    Empty shells (no voxels at that radius) are returned as NaN, not 0, so they do
    not get scored as zero correlation downstream.
    """