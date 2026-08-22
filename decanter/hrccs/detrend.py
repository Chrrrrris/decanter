"""Continuum preparation and fixed per-order SVD operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import percentile_filter


@dataclass(frozen=True)
class SVDPath:
    prepared: np.ndarray
    lower: dict[int, np.ndarray]
    residuals: dict[int, np.ndarray]
    u: np.ndarray
    valid: np.ndarray


def continuum_normalize(flux: np.ndarray, percentile: float, window: int) -> np.ndarray:
    window = max(5, int(window) | 1)
    filled = np.asarray(flux, dtype=float).copy()
    finite = np.isfinite(filled)
    if np.count_nonzero(finite) < 10:
        return np.full_like(filled, np.nan)
    x = np.arange(filled.size)
    filled[~finite] = np.interp(x[~finite], x[finite], filled[finite])
    continuum = percentile_filter(filled, percentile=percentile, size=window, mode="nearest")
    result = filled / np.clip(continuum, np.nanmedian(continuum) * 1.0e-4, None)
    result[~finite] = np.nan
    return result


def prepare_cube(flux: np.ndarray, percentile: float, window: int) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype=float)
    for i in range(flux.shape[0]):
        for j in range(flux.shape[1]):
            out[i, j] = continuum_normalize(flux[i, j], percentile, window)
    return out


def svd_path(matrix: np.ndarray, counts: tuple[int, ...], pixel_mask: np.ndarray,
             *, mode: str = "notebook") -> SVDPath:
    values = np.asarray(matrix, dtype=float)
    valid = np.isfinite(values) & pixel_mask[None, :]
    if mode == "notebook":
        usable_rows = np.any(valid, axis=1)
        usable_columns = pixel_mask & np.any(valid[usable_rows], axis=0)
    else:
        usable_rows = np.ones(values.shape[0], dtype=bool)
        usable_columns = pixel_mask & (
            np.sum(np.isfinite(values), axis=0) >= max(3, values.shape[0] // 2)
        )
    if np.count_nonzero(usable_columns) < 3:
        raise ValueError("too few valid pixels for SVD")
    local = values[np.ix_(usable_rows, usable_columns)]
    column_fill = np.nanmedian(local, axis=0)
    if mode == "notebook":
        global_fill = float(np.nanmedian(local))
        column_fill = np.where(np.isfinite(column_fill), column_fill, global_fill)
    prepared = np.where(np.isfinite(local), local, column_fill[None, :])
    if mode == "projected_log":
        prepared = np.log(np.clip(prepared, 1.0e-6, None))
        prepared -= np.nanmedian(prepared, axis=0, keepdims=True)
    elif mode != "notebook":
        raise ValueError(f"unknown SVD mode {mode!r}")
    u, singular, vt = np.linalg.svd(prepared, full_matrices=False)
    lowers: dict[int, np.ndarray] = {}
    residuals: dict[int, np.ndarray] = {}
    full_prepared = np.full_like(values, np.nan)
    full_prepared[np.ix_(usable_rows, usable_columns)] = prepared
    for count in sorted(set(counts)):
        count = min(int(count), u.shape[1])
        lower = (u[:, :count] * singular[:count]) @ vt[:count] if count else 0.0
        local_residual = prepared - lower
        full_lower = np.full_like(values, np.nan)
        full = np.full_like(values, np.nan)
        full_lower[np.ix_(usable_rows, usable_columns)] = lower
        full[np.ix_(usable_rows, usable_columns)] = local_residual
        full_lower[~valid] = np.nan
        full[~valid] = np.nan
        lowers[count] = full_lower
        residuals[count] = full
    return SVDPath(full_prepared, lowers, residuals, u, valid)


def apply_time_projection(model: np.ndarray, u: np.ndarray, count: int) -> np.ndarray:
    """Apply the fixed data-derived SVD time projection to a model cube."""
    values = np.asarray(model, dtype=float)
    filled = np.nan_to_num(values, nan=0.0)
    count = min(int(count), u.shape[1])
    filtered = filled - u[:, :count] @ (u[:, :count].T @ filled)
    filtered[~np.isfinite(values)] = np.nan
    return filtered
