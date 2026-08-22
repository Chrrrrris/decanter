#!/usr/bin/env python3
"""Independent notebook-style check of a packaged HRCCS H2O result.

The reference calculation intentionally does not import ``hrccs.analysis`` or
``hrccs.detrend``.  It shares the calibrated input loader, orbit metadata, and
the cached ExoJAX atmosphere with the production run, then independently
repeats the linear SVD, exact injected-template refit, fixed-interior Pearson
CCF, equal-order sum, and Kp--Vsys map from the WASP-69b notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decanter.hrccs.config import load_config
from decanter.hrccs.io import load_decanter
from decanter.hrccs.models import TemplateFactory
from decanter.hrccs.orbit import build_orbit

C_KMS = 299_792.458


def doppler_factor(velocity_kms):
    beta = np.asarray(velocity_kms, dtype=float) / C_KMS
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def notebook_svd(matrix, count):
    values = np.asarray(matrix, dtype=float)
    finite = np.isfinite(values)
    usable_rows = np.any(finite, axis=1)
    usable_columns = np.any(finite[usable_rows], axis=0)
    working = values[np.ix_(usable_rows, usable_columns)]
    working_finite = finite[np.ix_(usable_rows, usable_columns)]
    column_fill = np.nanmedian(working, axis=0)
    global_fill = float(np.nanmedian(working))
    column_fill = np.where(np.isfinite(column_fill), column_fill, global_fill)
    prepared = np.where(working_finite, working, column_fill[None, :])
    u_matrix, singular_values, vt_matrix = np.linalg.svd(prepared, full_matrices=False)
    lower_working = ((u_matrix[:, :count] * singular_values[:count])
                     @ vt_matrix[:count])
    residual_working = prepared - lower_working
    lower = np.full_like(values, np.nan)
    residual = np.full_like(values, np.nan)
    lower[np.ix_(usable_rows, usable_columns)] = np.where(
        working_finite, lower_working, np.nan
    )
    residual[np.ix_(usable_rows, usable_columns)] = np.where(
        working_finite, residual_working, np.nan
    )
    return lower, residual


def fixed_mask_pearson(data_row, model_rows, mask):
    x = np.asarray(data_row, dtype=float)[mask]
    y = np.asarray(model_rows, dtype=float)[:, mask]
    x = x - np.mean(x)
    y = y - np.mean(y, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=1))
    return np.divide(
        y @ x, denominator, out=np.full(model_rows.shape[0], np.nan),
        where=denominator > 0.0,
    )


def sigma_clipped_map(raw, sigma=3.0, iterations=10):
    finite = np.isfinite(raw)
    retained = finite.copy()
    for _ in range(iterations):
        values = raw[retained]
        center = float(np.median(values))
        scatter = float(np.std(values))
        updated = finite & (np.abs(raw - center) <= sigma * scatter)
        if np.array_equal(updated, retained):
            break
        retained = updated
    center = float(np.mean(raw[retained]))
    scatter = float(np.std(raw[retained]))
    return (raw - center) / scatter


def fixed_mask(cube, orbit, reduction, rv_grid, resolving_power):
    transmission = cube.telluric_transmission[orbit.in_transit]
    finite = np.isfinite(transmission)
    minimum = np.min(np.where(finite, transmission, np.inf), axis=0)
    minimum[~np.any(finite, axis=0)] = np.nan
    keep = np.isfinite(minimum) & (minimum >= reduction.telluric_threshold)
    for order, wave in enumerate(cube.wavelength_um):
        dv = C_KMS * np.nanmedian(np.diff(np.log(wave)))
        velocity_margin = (
            np.max(np.abs(rv_grid))
            + reduction.ccf_lsf_margin_widths * C_KMS / resolving_power
        )
        margin = reduction.edge_trim_pixels + int(np.ceil(velocity_margin / abs(dv)))
        keep[order, :margin] = False
        keep[order, -margin:] = False
    return keep


def moving_model(wavelength, wide_wave, wide_signal, phase, berv, kp, vsys, in_transit):
    result = np.zeros((phase.size,) + wavelength.shape, dtype=float)
    velocity = kp * np.sin(2.0 * np.pi * phase) + vsys - berv
    for frame in np.where(in_transit)[0]:
        factor = doppler_factor(velocity[frame])
        for order, wave in enumerate(wavelength):
            result[frame, order] = np.interp(
                wave / factor, wide_wave, wide_signal, left=np.nan, right=np.nan
            )
    return result


def reference_map(flux, wavelength, keep, model, phase, in_transit, rv_grid,
                  kp_grid, vsys_grid, expected_kp, expected_vsys, count):
    residuals, filtered = [], []
    for order in range(flux.shape[1]):
        lower, residual = notebook_svd(flux[:, order], count)
        injected = lower * (1.0 + model[:, order])
        _, injected_residual = notebook_svd(injected, count)
        _, control_residual = notebook_svd(lower, count)
        residuals.append(residual)
        filtered.append(injected_residual - control_residual)
    residuals = np.stack(residuals, axis=1)
    filtered = np.stack(filtered, axis=1)

    order_ccf = np.full((flux.shape[1], flux.shape[0], rv_grid.size), np.nan)
    for order, wave in enumerate(wavelength):
        mask = keep[order]
        for frame in np.where(in_transit)[0]:
            shifted = np.asarray([
                np.interp(
                    wave / doppler_factor(lag), wave, filtered[frame, order],
                    left=np.nan, right=np.nan,
                )
                for lag in rv_grid
            ])
            if (np.all(np.isfinite(residuals[frame, order, mask]))
                    and np.all(np.isfinite(shifted[:, mask]))):
                order_ccf[order, frame] = fixed_mask_pearson(
                    residuals[frame, order], shifted, mask
                )
    selected = np.where(keep.any(axis=1)[:, None, None], order_ccf, np.nan)
    combined = np.nansum(selected, axis=0)
    combined[~np.any(np.isfinite(selected), axis=0)] = np.nan

    use = np.asarray(in_transit, dtype=bool)
    sine = np.sin(2.0 * np.pi * phase[use])
    raw = np.full((kp_grid.size, vsys_grid.size), np.nan)
    for row, kp in enumerate(kp_grid):
        orbital = (kp - expected_kp) * sine
        for column, vsys in enumerate(vsys_grid):
            lag = orbital + vsys - expected_vsys
            values = np.asarray([
                np.interp(value, rv_grid, combined[index], left=np.nan, right=np.nan)
                for value, index in zip(lag, np.where(use)[0])
            ])
            raw[row, column] = np.mean(values)
    return sigma_clipped_map(raw), residuals, filtered, combined


def summary(values, kp_grid, vsys_grid, expected_kp, expected_vsys,
            kp_half_width, vsys_half_width):
    expected_index = (
        int(np.argmin(np.abs(kp_grid - expected_kp))),
        int(np.argmin(np.abs(vsys_grid - expected_vsys))),
    )
    local = ((np.abs(kp_grid[:, None] - expected_kp) <= kp_half_width)
             & (np.abs(vsys_grid[None, :] - expected_vsys) <= vsys_half_width))
    local_index = np.unravel_index(np.nanargmax(np.where(local, values, np.nan)), values.shape)
    return {
        "expected_snr": float(values[expected_index]),
        "local_peak_snr": float(values[local_index]),
        "local_peak_kp_kms": float(kp_grid[local_index[0]]),
        "local_peak_vsys_kms": float(vsys_grid[local_index[1]]),
    }


def comparison_pdf(path, packaged, reference, kp_grid, vsys_grid, expected_kp,
                   expected_vsys, metrics):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), constrained_layout=True)
    span = float(np.nanmax(np.abs(np.concatenate([packaged.ravel(), reference.ravel()]))))
    for axis, values, title in (
        (axes[0, 0], packaged, "Packaged HRCCS"),
        (axes[0, 1], reference, "Independent notebook reference"),
    ):
        image = axis.pcolormesh(vsys_grid, kp_grid, values, shading="auto",
                                cmap="RdBu_r", vmin=-span, vmax=span)
        axis.plot(expected_vsys, expected_kp, "+", color="k", ms=13, mew=2)
        axis.set(title=title, xlabel=r"$V_{sys}$ (km s$^{-1}$)",
                 ylabel=r"$K_p$ (km s$^{-1}$)")
        figure.colorbar(image, ax=axis, label="Map S/N")
    difference = packaged - reference
    diff_span = max(float(np.nanmax(np.abs(difference))), 1.0e-12)
    image = axes[1, 0].pcolormesh(
        vsys_grid, kp_grid, difference, shading="auto", cmap="RdBu_r",
        vmin=-diff_span, vmax=diff_span,
    )
    axes[1, 0].set(title="Packaged - reference", xlabel=r"$V_{sys}$ (km s$^{-1}$)",
                   ylabel=r"$K_p$ (km s$^{-1}$)")
    figure.colorbar(image, ax=axes[1, 0], label="S/N difference")

    row = int(np.argmin(np.abs(kp_grid - expected_kp)))
    axes[1, 1].plot(vsys_grid, packaged[row], label="Packaged", lw=1.7)
    axes[1, 1].plot(vsys_grid, reference[row], "--", label="Reference", lw=1.5)
    axes[1, 1].axvline(expected_vsys, color="0.3", ls=":")
    axes[1, 1].set(
        title=f"Expected-Kp slice ({kp_grid[row]:.0f} km s$^{{-1}}$)",
        xlabel=r"$V_{sys}$ (km s$^{-1}$)", ylabel="Map S/N",
    )
    axes[1, 1].legend(frameon=False)
    figure.suptitle(
        "WASP-69b H2O apples-to-apples validation\n"
        f"map r={metrics['map_correlation']:.12f}; "
        f"max |difference|={metrics['max_abs_snr_difference']:.3e}",
        fontsize=14,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--species", default="H2O")
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.reduction.analysis_mode != "notebook":
        raise ValueError("comparison requires reduction.analysis_mode='notebook'")
    if config.reduction.order_combination != "equal":
        raise ValueError("comparison requires reduction.order_combination='equal'")
    if config.reduction.template_signal != "absolute_depth":
        raise ValueError("comparison requires reduction.template_signal='absolute_depth'")
    count = int(config.reduction.svd_components[0])
    if len(config.reduction.svd_components) != 1:
        raise ValueError("comparison config must specify exactly one SVD count")

    cube = load_decanter(
        config.input.decanter_dir, fsr_cut=config.input.fsr_cut,
        orders=config.input.orders, telluric_product=config.input.telluric_product,
    )
    orbit = build_orbit(
        config.system, cube.time_jd_utc, cube.metadata,
        Path(config.atmosphere.cache_dir).expanduser() / "simbad",
    )
    rv_grid, kp_grid, vsys_grid = config.grids(orbit.stellar_rv_kms)
    resolving_power = config.atmosphere.resolving_power_for(cube.instmode)
    keep = fixed_mask(cube, orbit, config.reduction, rv_grid, resolving_power)
    retained = np.sum(keep, axis=1) >= config.reduction.min_valid_pixels
    wavelength = cube.wavelength_um[retained]
    flux = cube.flux[:, retained]
    keep = keep[retained]

    from dataclasses import replace
    atmosphere = replace(config.atmosphere, resolving_power=resolving_power)
    factory = TemplateFactory(config.system, atmosphere, instmode=cube.instmode)
    wide = factory.build_wide(args.species, wavelength, show_progress=True)
    model = moving_model(
        wavelength, wide.wavelength_um, -wide.transit_depth,
        orbit.phase, orbit.berv_kms, config.system.expected_kp_kms,
        orbit.stellar_rv_kms, orbit.in_transit,
    )
    reference, _, _, _ = reference_map(
        flux, wavelength, keep, model, orbit.phase, orbit.in_transit,
        rv_grid, kp_grid, vsys_grid, config.system.expected_kp_kms,
        orbit.stellar_rv_kms, count,
    )

    product = Path(config.output_dir).expanduser().resolve() / f"{args.species}_hrccs_products.npz"
    with np.load(product, allow_pickle=False) as loaded:
        packaged = np.asarray(loaded["observed_snr_map"], dtype=float)
        packaged_mask = np.asarray(loaded["telluric_keep_mask"], dtype=bool)
    if not np.array_equal(packaged_mask, keep):
        raise RuntimeError("packaged and reference masks differ")
    difference = packaged - reference
    metrics = {
        "schema": "decanter.hrccs.notebook_comparison.v1",
        "species": args.species,
        "svd_components": count,
        "map_correlation": float(np.corrcoef(packaged.ravel(), reference.ravel())[0, 1]),
        "max_abs_snr_difference": float(np.nanmax(np.abs(difference))),
        "rms_snr_difference": float(np.sqrt(np.nanmean(difference**2))),
        "packaged": summary(
            packaged, kp_grid, vsys_grid, config.system.expected_kp_kms,
            orbit.stellar_rv_kms, config.search.local_kp_half_width_kms,
            config.search.local_vsys_half_width_kms,
        ),
        "notebook_reference": summary(
            reference, kp_grid, vsys_grid, config.system.expected_kp_kms,
            orbit.stellar_rv_kms, config.search.local_kp_half_width_kms,
            config.search.local_vsys_half_width_kms,
        ),
        "shared_inputs": {
            "calibrated_directory": str(Path(config.input.decanter_dir).resolve()),
            "telluric_threshold": config.reduction.telluric_threshold,
            "orders": cube.orders[retained].tolist(),
            "instrument_resolution": resolving_power,
            "template_cache": str(factory._path(args.species, wide.wavelength_um)),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "wasp69b_h2o_notebook_pipeline_comparison.json"
    pdf_path = args.output_dir / "wasp69b_h2o_notebook_pipeline_comparison.pdf"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    comparison_pdf(
        pdf_path, packaged, reference, kp_grid, vsys_grid,
        config.system.expected_kp_kms, orbit.stellar_rv_kms, metrics,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {pdf_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
