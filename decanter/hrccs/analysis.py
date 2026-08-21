"""SVD-filtered Pearson CCFs, Kp--Vsys maps, and injection recovery."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from decanter.hrccs.detrend import SVDPath, apply_time_projection, svd_path

C_KMS = 299_792.458


@dataclass(frozen=True)
class ComponentResult:
    count: int
    residual_cube: np.ndarray
    filtered_template_cube: np.ndarray
    exposure_ccf: np.ndarray
    raw_map: np.ndarray
    snr_map: np.ndarray
    expected_snr: float
    local_peak_snr: float
    local_peak_kp_kms: float
    local_peak_vsys_kms: float
    peak_snr: float
    peak_kp_kms: float
    peak_vsys_kms: float


@dataclass(frozen=True)
class SpeciesResult:
    species: str
    raw_templates: np.ndarray
    planet_model_cube: np.ndarray
    mask: np.ndarray
    svd_paths: tuple[SVDPath, ...]
    component_results: tuple[ComponentResult, ...]
    selected: ComponentResult
    injected: ComponentResult
    null_mean_map: np.ndarray
    null_snr_at_expected: np.ndarray
    null_local_peak_snr: np.ndarray
    null_false_alarm_fraction: float


def relativistic_factor(velocity_kms):
    beta = np.asarray(velocity_kms, dtype=float) / C_KMS
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def shift_template(wave_um, template, velocity_kms):
    sample = np.asarray(wave_um) / relativistic_factor(velocity_kms)
    return np.interp(sample, wave_um, template, left=np.nan, right=np.nan)


def planet_model_cube(wavelength_um, templates, phase, berv_kms, kp_kms, vsys_kms,
                      transit_weight, scale=1.0):
    n_frames = phase.size
    out = np.full((n_frames,) + templates.shape, np.nan)
    velocity = kp_kms * np.sin(2.0 * np.pi * phase) + vsys_kms - berv_kms
    for i in range(n_frames):
        for j in range(templates.shape[0]):
            out[i, j] = (scale * transit_weight[i]
                         * shift_template(wavelength_um[j], templates[j], velocity[i]))
    return out


def fixed_pearson_ccf(data, model, wave, rv_grid, fixed_mask):
    output = np.full((data.shape[0], rv_grid.size), np.nan)
    for i in range(data.shape[0]):
        mask = fixed_mask & np.isfinite(data[i]) & np.isfinite(model[i])
        if np.count_nonzero(mask) < 3:
            continue
        bank = np.asarray([shift_template(wave, model[i], velocity) for velocity in rv_grid])
        mask &= np.all(np.isfinite(bank), axis=0)
        if np.count_nonzero(mask) < 3:
            continue
        x = data[i, mask] - np.mean(data[i, mask])
        y = bank[:, mask] - np.mean(bank[:, mask], axis=1, keepdims=True)
        denom = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=1))
        output[i] = np.divide(y @ x, denom, out=np.full(rv_grid.size, np.nan), where=denom > 0)
    return output


def combine_order_ccfs(order_ccf, information):
    weights = np.sqrt(np.maximum(np.asarray(information, dtype=float), 0.0))
    weights[~np.isfinite(weights)] = 0.0
    if not np.any(weights > 0):
        weights[:] = 1.0
    weights /= weights.sum()
    valid = np.isfinite(order_ccf)
    numerator = np.nansum(order_ccf * weights[:, None, None], axis=0)
    denominator = np.sum(valid * weights[:, None, None], axis=0)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def kp_vsys_map(exposure_ccf, rv_grid, phase, transit_weight, kp_grid, vsys_grid,
                expected_kp, expected_vsys):
    use = (transit_weight > 0) & np.any(np.isfinite(exposure_ccf), axis=1)
    result = np.full((kp_grid.size, vsys_grid.size), np.nan)
    sine = np.sin(2.0 * np.pi * phase[use])
    weights = transit_weight[use]
    for row, kp in enumerate(kp_grid):
        orbital = (kp - expected_kp) * sine
        for col, vsys in enumerate(vsys_grid):
            lag = orbital + vsys - expected_vsys
            values = np.asarray([
                np.interp(value, rv_grid, exposure_ccf[index], left=np.nan, right=np.nan)
                for value, index in zip(lag, np.where(use)[0])
            ])
            finite = np.isfinite(values)
            if np.any(finite):
                result[row, col] = np.sum(weights[finite] * values[finite]) / np.sqrt(
                    np.sum(weights[finite] ** 2)
                )
    return result


def standardize_map(raw, sigma=3.0, iterations=10):
    finite = np.isfinite(raw)
    keep = finite.copy()
    for _ in range(iterations):
        values = raw[keep]
        if values.size < 2:
            break
        center, scatter = np.median(values), np.std(values)
        if not np.isfinite(scatter) or scatter <= 0:
            break
        updated = finite & (np.abs(raw - center) <= sigma * scatter)
        if np.array_equal(updated, keep):
            break
        keep = updated
    center = np.mean(raw[keep]) if np.any(keep) else 0.0
    scatter = np.std(raw[keep]) if np.any(keep) else 1.0
    return (raw - center) / max(float(scatter), 1.0e-12)


def _map_summary(snr_map, kp_grid, vsys_grid, expected_kp, expected_vsys,
                 local_kp_half_width, local_vsys_half_width):
    expected = float(snr_map[np.argmin(abs(kp_grid - expected_kp)),
                             np.argmin(abs(vsys_grid - expected_vsys))])
    local_mask = ((np.abs(kp_grid[:, None] - expected_kp) <= local_kp_half_width)
                  & (np.abs(vsys_grid[None, :] - expected_vsys) <= local_vsys_half_width))
    local_values = np.where(local_mask, snr_map, np.nan)
    if np.any(np.isfinite(local_values)):
        local = np.unravel_index(np.nanargmax(local_values), local_values.shape)
        local_result = (float(local_values[local]), float(kp_grid[local[0]]),
                        float(vsys_grid[local[1]]))
    else:
        local_result = (np.nan, np.nan, np.nan)
    if np.any(np.isfinite(snr_map)):
        peak = np.unravel_index(np.nanargmax(snr_map), snr_map.shape)
        return (expected, *local_result, float(snr_map[peak]),
                float(kp_grid[peak[0]]), float(vsys_grid[peak[1]]))
    return expected, *local_result, np.nan, np.nan, np.nan


def evaluate(count, residual_cube, filtered_model, wavelength_um, mask, phase,
             transit_weight, rv_grid, kp_grid, vsys_grid, expected_kp, expected_vsys,
             sigma_clip, local_kp_half_width, local_vsys_half_width):
    order_ccf = []
    information = []
    for order in range(residual_cube.shape[1]):
        order_ccf.append(fixed_pearson_ccf(
            residual_cube[:, order], filtered_model[:, order], wavelength_um[order],
            rv_grid, mask[order],
        ))
        information.append(np.nansum(np.where(mask[order][None, :],
                                              filtered_model[:, order], np.nan) ** 2))
    combined = combine_order_ccfs(np.asarray(order_ccf), information)
    raw = kp_vsys_map(combined, rv_grid, phase, transit_weight, kp_grid, vsys_grid,
                      expected_kp, expected_vsys)
    snr = standardize_map(raw, sigma=sigma_clip)
    expected, local_peak, local_kp, local_vsys, peak, peak_kp, peak_vsys = _map_summary(
        snr, kp_grid, vsys_grid, expected_kp, expected_vsys,
        local_kp_half_width, local_vsys_half_width,
    )
    return ComponentResult(count, residual_cube, filtered_model, combined, raw, snr,
                           expected, local_peak, local_kp, local_vsys,
                           peak, peak_kp, peak_vsys)


def _paths(prepared, mask, counts):
    paths = []
    for order in range(prepared.shape[1]):
        paths.append(svd_path(prepared[:, order], counts, mask[order]))
    return tuple(paths)


def _residual_cube(paths, count):
    return np.stack([path.residuals[min(count, max(path.residuals))] for path in paths], axis=1)


def _filtered_cube(model, paths, count):
    return np.stack([apply_time_projection(model[:, order], path.u, count)
                     for order, path in enumerate(paths)], axis=1)


def _select_component(components):
    """Select by the finite local maximum near the expected planet location."""
    finite = [item for item in components if np.isfinite(item.local_peak_snr)]
    if not finite:
        raise ValueError("all SVD-component local-maximum S/N values are non-finite")
    return max(finite, key=lambda item: item.local_peak_snr)


def run_species(species, prepared, wavelength_um, raw_templates, mask, phase, berv_kms,
                transit_weight, rv_grid, kp_grid, vsys_grid, expected_kp, expected_vsys,
                counts, sigma_clip, local_kp_half_width, local_vsys_half_width,
                injection_scale, seed, null_realizations):
    expected_model = planet_model_cube(
        wavelength_um, raw_templates, phase, berv_kms, expected_kp, expected_vsys,
        transit_weight, scale=injection_scale,
    )
    paths = _paths(prepared, mask, counts)
    components = []
    for count in counts:
        components.append(evaluate(
            count, _residual_cube(paths, count), _filtered_cube(expected_model, paths, count),
            wavelength_um, mask, phase, transit_weight, rv_grid, kp_grid, vsys_grid,
            expected_kp, expected_vsys, sigma_clip,
            local_kp_half_width, local_vsys_half_width,
        ))
    selected = _select_component(components)

    oot = transit_weight <= 0
    if np.count_nonzero(oot) < 2:
        raise ValueError("injection recovery needs at least two out-of-transit exposures")
    # Fully masked edge pixels are expected; avoid emitting one warning per order.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        oot_center = np.nanmedian(selected.residual_cube[oot], axis=0)
        noise_sigma = 1.4826 * np.nanmedian(
            np.abs(selected.residual_cube[oot] - oot_center[None, :]), axis=0
        )
    for order in range(noise_sigma.shape[0]):
        good = np.isfinite(noise_sigma[order]) & (noise_sigma[order] > 0)
        fill = np.nanmedian(noise_sigma[order, good]) if np.any(good) else 1.0e-3
        noise_sigma[order, ~good] = fill
    baseline = np.nanmedian(prepared[oot], axis=0)
    def synthetic(model, local_seed):
        local_rng = np.random.default_rng(local_seed)
        return baseline[None, :, :] + model + local_rng.normal(
            0.0, noise_sigma[None, :, :], size=prepared.shape
        )

    injected_data = synthetic(expected_model, seed)
    injected_paths = _paths(injected_data, mask, (selected.count,))
    injected = evaluate(
        selected.count, _residual_cube(injected_paths, selected.count),
        _filtered_cube(expected_model, injected_paths, selected.count), wavelength_um, mask,
        phase, transit_weight, rv_grid, kp_grid, vsys_grid, expected_kp, expected_vsys, sigma_clip,
        local_kp_half_width, local_vsys_half_width,
    )
    null_maps, null_expected, null_local = [], [], []
    for index in range(null_realizations):
        null_data = synthetic(np.zeros_like(expected_model), seed + 1000 + index)
        null_paths = _paths(null_data, mask, (selected.count,))
        null = evaluate(
            selected.count, _residual_cube(null_paths, selected.count),
            _filtered_cube(expected_model, null_paths, selected.count), wavelength_um, mask,
            phase, transit_weight, rv_grid, kp_grid, vsys_grid,
            expected_kp, expected_vsys, sigma_clip,
            local_kp_half_width, local_vsys_half_width,
        )
        # Match the injection recovery: every null uses the rank selected from
        # the observed data, without re-tuning on the null realization.
        null_maps.append(null.snr_map)
        null_expected.append(null.expected_snr)
        null_local.append(null.local_peak_snr)
    null_expected = np.asarray(null_expected)
    null_local = np.asarray(null_local)
    fap = float((1 + np.count_nonzero(null_local >= selected.local_peak_snr)) /
                (1 + null_local.size))
    return SpeciesResult(species, raw_templates, expected_model, mask, paths,
                         tuple(components), selected, injected,
                         np.nanmean(null_maps, axis=0), null_expected, null_local, fap)
