#!/usr/bin/env python3
"""Standalone notebook-method injection test for the WASP-69b JWST spectrum.

This intentionally bypasses ``TemplateFactory`` and ExoJAX.  It reads the
project's high-resolution evening-limb spectrum, matches its declared R=30,000
LSF to the Decanter product's instrumental resolution, and otherwise uses the
same data, mask, orbit, SVD, CCF, and Kp--Vsys conventions as ``decanter-hrccs``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from decanter.hrccs.analysis import (
    _filtered_cube,
    _paths,
    _residual_cube,
    _select_component,
    evaluate,
    planet_model_cube,
)
from decanter.hrccs.config import load_config
from decanter.hrccs.io import load_decanter
from decanter.hrccs.orbit import build_orbit
from decanter.hrccs.pipeline import _mask

C_KMS = 299_792.458
SOURCE_RESOLVING_POWER = 30_000.0


def load_jwst_template(path: Path, target_resolution: float):
    table = np.genfromtxt(path, delimiter=",", names=True)
    required = {"wavelength_micron", "transit_depth"}
    if not required.issubset(table.dtype.names or ()):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    wave = np.asarray(table["wavelength_micron"], dtype=float)
    depth = np.asarray(table["transit_depth"], dtype=float)
    finite = np.isfinite(wave) & np.isfinite(depth)
    wave, depth = wave[finite], depth[finite]
    order = np.argsort(wave)
    wave, depth = wave[order], depth[order]
    unique = np.concatenate(([True], np.diff(wave) > 0.0))
    wave, depth = wave[unique], depth[unique]
    if wave.size < 100 or np.any(np.diff(wave) <= 0.0):
        raise ValueError("JWST-derived template wavelength grid is invalid")
    if target_resolution > SOURCE_RESOLVING_POWER:
        raise ValueError(
            f"cannot sharpen R={SOURCE_RESOLVING_POWER:.0f} source to "
            f"R={target_resolution:.0f}"
        )

    pixel_velocity = C_KMS * float(np.median(np.diff(np.log(wave))))
    convolution_fwhm = np.sqrt(
        (C_KMS / target_resolution) ** 2
        - (C_KMS / SOURCE_RESOLVING_POWER) ** 2
    )
    sigma_pixels = convolution_fwhm / 2.354820045 / pixel_velocity
    matched_depth = gaussian_filter1d(depth, sigma_pixels, mode="nearest")
    return wave, depth, matched_depth, float(sigma_pixels)


def noise_model(prepared, residual_cube, transit_weight):
    oot = np.asarray(transit_weight) <= 0.0
    if np.count_nonzero(oot) < 2:
        raise ValueError("injection recovery requires at least two OOT exposures")
    center = np.nanmedian(residual_cube[oot], axis=0)
    sigma = 1.4826 * np.nanmedian(
        np.abs(residual_cube[oot] - center[None, :]), axis=0
    )
    for order in range(sigma.shape[0]):
        good = np.isfinite(sigma[order]) & (sigma[order] > 0.0)
        fill = np.nanmedian(sigma[order, good]) if np.any(good) else 1.0e-3
        sigma[order, ~good] = fill
    return np.nanmedian(prepared[oot], axis=0), sigma


def plot_result(path, wide_wave, native_depth, matched_depth, injected, kp_grid,
                vsys_grid, expected_kp, expected_vsys):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].plot(wide_wave, native_depth * 1.0e6, lw=0.7, alpha=0.65,
                 label="source R=30,000")
    axes[0].plot(wide_wave, matched_depth * 1.0e6, lw=0.9,
                 label="WINERED R=28,000")
    axes[0].set(xlabel=r"Vacuum wavelength ($\mu$m)",
                ylabel="Absolute transit depth (ppm)",
                title="JWST-derived evening-limb spectrum")
    axes[0].legend(frameon=False)

    span = float(np.nanpercentile(np.abs(injected.snr_map), 99.5))
    image = axes[1].pcolormesh(vsys_grid, kp_grid, injected.snr_map,
                               shading="auto", cmap="RdBu_r",
                               vmin=-span, vmax=span)
    axes[1].plot(expected_vsys, expected_kp, "+", color="k", ms=13, mew=2)
    axes[1].plot(injected.local_peak_vsys_kms, injected.local_peak_kp_kms,
                 "o", mfc="none", mec="k", ms=8)
    axes[1].set(xlabel=r"$V_{sys}$ (km s$^{-1}$)",
                ylabel=r"$K_p$ (km s$^{-1}$)",
                title=(f"Synthetic recovery: expected {injected.expected_snr:+.2f}$\sigma$\n"
                       f"local max {injected.local_peak_snr:+.2f}$\sigma$"))
    figure.colorbar(image, ax=axes[1], label="Map S/N")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="examples/hrccs/wasp69b_matched_validation.toml",
        help="WASP-69b HRCCS configuration (default: notebook validation profile)",
    )
    parser.add_argument(
        "--template", default="../wasp69b_evening_median_H2O_R30000_0p8-1p5um.csv",
        help="JWST-derived absolute transit-depth CSV",
    )
    parser.add_argument(
        "--out", default="../outputs/hrccs_validation/wasp69b_jwst_injection",
        help="Output directory",
    )
    parser.add_argument(
        "--telluric-threshold", type=float,
        help="Override the configuration's minimum retained transmission",
    )
    parser.add_argument(
        "--svd-components",
        help="Override searched ranks with a comma-separated list, e.g. 1,2,...,12",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    reduction_overrides = {}
    if args.telluric_threshold is not None:
        reduction_overrides["telluric_threshold"] = args.telluric_threshold
    if args.svd_components:
        reduction_overrides["svd_components"] = tuple(
            int(value) for value in args.svd_components.split(",")
        )
    if reduction_overrides:
        config = replace(
            config, reduction=replace(config.reduction, **reduction_overrides)
        )
    config.validate()
    if config.reduction.analysis_mode != "notebook":
        raise ValueError("this standalone test requires analysis_mode='notebook'")
    if config.reduction.template_signal != "absolute_depth":
        raise ValueError("this standalone test requires template_signal='absolute_depth'")

    cube = load_decanter(
        config.input.decanter_dir, fsr_cut=config.input.fsr_cut,
        orders=config.input.orders, telluric_product=config.input.telluric_product,
    )
    orbit = build_orbit(
        config.system, cube.time_jd_utc, cube.metadata,
        Path(config.atmosphere.cache_dir).expanduser() / "simbad",
    )
    rv_grid, kp_grid, vsys_grid = config.grids(orbit.stellar_rv_kms)
    resolution = config.atmosphere.resolving_power_for(cube.instmode)
    keep = _mask(cube, config.reduction, rv_grid, orbit.in_transit, resolution)
    retained = np.sum(keep, axis=1) >= config.reduction.min_valid_pixels
    wavelength = cube.wavelength_um[retained]
    prepared = np.asarray(cube.flux[:, retained], dtype=float).copy()
    keep = keep[retained]

    wide_wave, native_depth, matched_depth, sigma_pixels = load_jwst_template(
        Path(args.template).expanduser().resolve(), resolution
    )
    if (wide_wave[0] > np.nanmin(wavelength)
            or wide_wave[-1] < np.nanmax(wavelength)):
        raise ValueError("JWST-derived spectrum does not span all retained orders")
    wide_signal = -matched_depth
    order_templates = np.asarray([
        np.interp(wave, wide_wave, wide_signal) for wave in wavelength
    ])
    expected_model = planet_model_cube(
        wavelength, order_templates, orbit.phase, orbit.berv_kms,
        config.system.expected_kp_kms, orbit.stellar_rv_kms,
        orbit.transit_weight, scale=config.injection.scale,
        wide_wavelength_um=wide_wave, wide_template=wide_signal,
    )

    counts = config.reduction.svd_components
    paths = _paths(prepared, keep, counts, "notebook")
    observed = []
    from tqdm.auto import tqdm

    for count in tqdm(counts, desc="JWST-template SVD ranks", unit="rank"):
        observed.append(evaluate(
            count, _residual_cube(paths, count),
            _filtered_cube(expected_model, paths, count, "notebook"),
            wavelength, keep, orbit.phase, orbit.transit_weight,
            rv_grid, kp_grid, vsys_grid, config.system.expected_kp_kms,
            orbit.stellar_rv_kms, config.search.map_sigma_clip,
            config.search.local_kp_half_width_kms,
            config.search.local_vsys_half_width_kms, "equal",
        ))
    selected = _select_component(observed)

    baseline, sigma = noise_model(
        prepared, selected.residual_cube, orbit.transit_weight
    )
    rng = np.random.default_rng(config.injection.random_seed)
    injected_data = (
        baseline[None, :, :] * (1.0 + expected_model)
        + rng.normal(0.0, sigma[None, :, :], size=prepared.shape)
    )
    print(f"Running one JWST-spectrum injection at selected rank {selected.count}...")
    injected_paths = _paths(injected_data, keep, (selected.count,), "notebook")
    injected = evaluate(
        selected.count, _residual_cube(injected_paths, selected.count),
        _filtered_cube(expected_model, injected_paths, selected.count, "notebook"),
        wavelength, keep, orbit.phase, orbit.transit_weight,
        rv_grid, kp_grid, vsys_grid, config.system.expected_kp_kms,
        orbit.stellar_rv_kms, config.search.map_sigma_clip,
        config.search.local_kp_half_width_kms,
        config.search.local_vsys_half_width_kms, "equal",
    )

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "decanter.hrccs.wasp69b_jwst_injection.v1",
        "source": str(Path(args.template).expanduser().resolve()),
        "source_description": "JWST-derived WASP-69b evening-limb absolute transit depth",
        "source_resolving_power": SOURCE_RESOLVING_POWER,
        "instrument_resolving_power": resolution,
        "convolution_sigma_source_pixels": sigma_pixels,
        "analysis_mode": "notebook",
        "telluric_threshold": config.reduction.telluric_threshold,
        "selected_svd_components": selected.count,
        "injection_scale": config.injection.scale,
        "injected_expected_snr": injected.expected_snr,
        "injected_local_peak_snr": injected.local_peak_snr,
        "injected_local_peak_kp_kms": injected.local_peak_kp_kms,
        "injected_local_peak_vsys_kms": injected.local_peak_vsys_kms,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.savez_compressed(
        output / "wasp69b_jwst_injection_products.npz",
        wide_wavelength_um=wide_wave, native_absolute_depth=native_depth,
        instrument_absolute_depth=matched_depth,
        kp_grid_kms=kp_grid, vsys_grid_kms=vsys_grid,
        injected_snr_map=injected.snr_map,
        selected_svd_components=np.asarray(selected.count),
        injected_expected_snr=np.asarray(injected.expected_snr),
        injected_local_peak_snr=np.asarray(injected.local_peak_snr),
    )
    plot_result(
        output / "wasp69b_jwst_injection.png", wide_wave, native_depth,
        matched_depth, injected, kp_grid, vsys_grid,
        config.system.expected_kp_kms, orbit.stellar_rv_kms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
