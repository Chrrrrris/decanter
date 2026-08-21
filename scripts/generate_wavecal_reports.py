#!/usr/bin/env python
"""Generate physical hybrid wavecal reports for the three local datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from decanter.wavecal import WavecalConfig, load_series
from decanter.wavecal.measure import (
    ccf_shifts,
    continuum_normalize,
    measure_series_shifts,
    robust_scatter,
    shift_grid_pixels,
)
from decanter.wavecal.report import wavecal_report_pdf
from decanter.wavecal.opacity import series_tau
from decanter.wavecal.solve import _refit_one, solve
from decanter.wavecal.solution import C_KMS
from decanter.wavecal.telluric import transmission_numpy

WORKSPACE = Path(
    os.environ.get("DECANTER_VALIDATION_ROOT", Path(__file__).resolve().parents[2])
)
DATASETS = {
    "toi2109b": ("TOI-2109b", WORKSPACE / "outputs/decanter_reductions/toi2109b"),
    "wasp69b": ("WASP-69b", WORKSPACE / "outputs/decanter_reductions/wasp69b"),
    "toi3486b": ("TOI-3486b", WORKSPACE / "outputs/decanter_reductions/toi3486b"),
}


def _post_correction_telluric_check(run):
    """Re-measure the physical telluric CCF after applying the final WCS.

    The fitted zero-shift template and its fixed interior support are held
    fixed.  Only the data cube is transformed to the corrected wavelength
    coordinates.  This mirrors the fixed-interior CCF logic in the science
    notebooks and tests the application/sign of the saved solution instead of
    merely restating its velocity array.
    """
    series = run.series
    corrected = np.full_like(series.obj, np.nan, dtype=float)
    for i in range(series.n_frames):
        for j in range(series.n_orders):
            velocity = float(run.solution.velocity[i, j])
            if not np.isfinite(velocity):
                continue
            corrected_wave = series.wave[:, j] / (1.0 + velocity / C_KMS)
            corrected[i, :, j] = np.interp(
                series.wave[:, j],
                corrected_wave,
                np.asarray(series.obj[i, :, j], dtype=float),
                left=np.nan,
                right=np.nan,
            )

    width = max(
        51,
        int(round(151 * 0.96 / float(np.median(series.dv_pix_kms)))) | 1,
    )
    signal = np.full_like(corrected, np.nan)
    for i in range(series.n_frames):
        for j in range(series.n_orders):
            signal[i, :, j] = 1.0 - continuum_normalize(corrected[i, :, j], width)
    residual_velocity, residual_peak = measure_series_shifts(
        signal,
        1.0 - run.telluric_model.native_template,
        run.telluric_support,
        series.dv_pix_kms,
        search_kms=4.0,
    )
    rich = run.telluric_model.rich_orders(run.config.telluric_rms_threshold)
    accepted = (
        run.telluric_accepted
        & np.isfinite(residual_velocity)
        & (residual_peak >= run.config.telluric_peak_threshold)
    )

    # The applied direct telluric values come from the nonlinear
    # per-exposure refit (when hybrid_refit is selected), not from the CCF
    # seed. Re-run that exact estimator at the corrected wavelength scale.
    # A correctly applied calibration must return zero shift within numerical
    # and interpolation precision.
    tau, _ = series_tau(
        series,
        tuple(run.config.species),
        run.config.linelist_dir,
        run.config.cache_dir,
    )
    normalized = 1.0 - signal
    refit_velocity = np.full_like(residual_velocity, np.nan)
    n_species = tau.shape[0]
    for j in np.where(rich)[0]:
        bound = 4.0 / series.dv_pix_kms[j]
        for i in np.where(run.telluric_accepted[:, j])[0]:
            seed = residual_velocity[i, j] / series.dv_pix_kms[j]
            if not np.isfinite(seed):
                seed = 0.0
            shift = _refit_one(
                tau[:, :, j],
                normalized[i, :, j],
                run.telluric_model.parameters[j],
                str(run.telluric_model.family[j]),
                n_species,
                series.n_pixels,
                seed,
                bound,
            )
            if np.isfinite(shift):
                refit_velocity[i, j] = shift * series.dv_pix_kms[j]
    refit_accepted = run.telluric_accepted & np.isfinite(refit_velocity)

    # Notebook-equivalent validation: each exposure is correlated against its
    # own zero-shift physical template (its fitted column scales, with the
    # order LSF frozen), on fixed line support.  This is the direct analogue
    # of ``residual_shift_against_oot_median_template`` in the referenced
    # WASP-69b notebook, including its +/-0.50 km/s pass criterion.
    notebook_velocity = np.full_like(residual_velocity, np.nan)
    notebook_peak = np.full_like(residual_peak, np.nan)
    if run.telluric_refit_parameters is not None:
        for j in np.where(rich)[0]:
            grid = shift_grid_pixels(
                10.0, float(series.dv_pix_kms[j]), step_kms=0.05
            )
            for i in np.where(run.telluric_accepted[:, j])[0]:
                fitted = run.telluric_refit_parameters[i, j]
                if not np.all(np.isfinite(fitted)):
                    continue
                parameters = np.asarray(
                    run.telluric_model.parameters[j], dtype=float
                ).copy()
                parameters[:n_species] = fitted[:n_species]
                template = transmission_numpy(
                    tau[:, :, j],
                    parameters,
                    str(run.telluric_model.family[j]),
                    n_species,
                    series.n_pixels,
                    shift=0.0,
                    with_continuum=False,
                )
                support = run.telluric_support[:, j] & (template < 0.997)
                shift, peak = ccf_shifts(
                    signal[i, :, j], 1.0 - template, support, grid
                )
                notebook_velocity[i, j] = shift[0] * series.dv_pix_kms[j]
                notebook_peak[i, j] = peak[0]
    notebook_accepted = run.telluric_accepted & np.isfinite(notebook_velocity)
    return (
        residual_velocity,
        residual_peak,
        accepted,
        refit_velocity,
        refit_accepted,
        notebook_velocity,
        notebook_peak,
        notebook_accepted,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()

    label, reduction_dir = DATASETS[args.dataset]
    print(f"Loading {label}: {reduction_dir}", flush=True)
    series = load_series(reduction_dir)
    print(series.summary(), flush=True)
    config = WavecalConfig(
        mode="auto",
        zero_point="absolute",
        assembly="ladder",
        linelist_dir=str(WORKSPACE),
        cache_dir=str(WORKSPACE / "outputs/wavecal_cache" / args.dataset),
    )
    run = solve(series, config, verbose=True, return_diagnostics=True)
    output = args.output_dir / f"{args.dataset}_wavecal_diagnostics.pdf"
    wavecal_report_pdf(run, output, dataset=label)
    solution_path = args.output_dir / f"{args.dataset}_wavecal_solution.npz"
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    run.solution.save_npz(solution_path)

    (
        residual_velocity,
        residual_peak,
        residual_accepted,
        residual_refit_velocity,
        residual_refit_accepted,
        notebook_velocity,
        notebook_peak,
        notebook_accepted,
    ) = (
        _post_correction_telluric_check(run)
    )
    residual_csv = args.output_dir / f"{args.dataset}_telluric_residual_drift.csv"
    with residual_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame_id", "order", "pre_telluric_kms",
                "applied_hybrid_kms", "post_telluric_kms", "post_ccf_peak",
                "post_accepted", "post_refit_telluric_kms",
                "post_refit_accepted",
                "notebook_validation_kms", "notebook_validation_peak",
                "notebook_validation_accepted",
            ),
        )
        writer.writeheader()
        for i, frame_id in enumerate(series.frame_ids):
            for j, order in enumerate(series.orders):
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "order": int(order),
                        "pre_telluric_kms": run.telluric_velocity[i, j],
                        "applied_hybrid_kms": run.solution.velocity[i, j],
                        "post_telluric_kms": residual_velocity[i, j],
                        "post_ccf_peak": residual_peak[i, j],
                        "post_accepted": bool(residual_accepted[i, j]),
                        "post_refit_telluric_kms": residual_refit_velocity[i, j],
                        "post_refit_accepted": bool(residual_refit_accepted[i, j]),
                        "notebook_validation_kms": notebook_velocity[i, j],
                        "notebook_validation_peak": notebook_peak[i, j],
                        "notebook_validation_accepted": bool(notebook_accepted[i, j]),
                    }
                )

    rich_t = run.telluric_model.rich_orders(config.telluric_rms_threshold)
    rich_o = run.oh_model.rich_orders(config.oh_rich_min_lines)
    paired = run.telluric_accepted & run.oh_accepted & (rich_t & rich_o)[None, :]
    residuals = []
    offsets = []
    for j in np.where(rich_t & rich_o)[0]:
        good = paired[:, j]
        if np.count_nonzero(good) < 2:
            continue
        tell = run.telluric_velocity[good, j]
        air = run.oh_velocity[good, j]
        offsets.append(float(np.nanmedian(air - tell) * 1e3))
        tell = tell - np.nanmedian(tell)
        air = air - np.nanmedian(air)
        residuals.extend(((air - tell) * 1e3).tolist())
    delta = np.asarray(residuals, dtype=float)
    post = residual_velocity[residual_accepted]
    post_refit = residual_refit_velocity[residual_refit_accepted]
    pre_common = np.nanmedian(
        np.where(run.telluric_accepted, run.telluric_velocity, np.nan), axis=1
    )
    post_common = np.nanmedian(
        np.where(residual_accepted, residual_velocity, np.nan), axis=1
    )
    post_refit_common = np.nanmedian(
        np.where(residual_refit_accepted, residual_refit_velocity, np.nan), axis=1
    )
    notebook_frame = np.nanmedian(
        np.where(notebook_accepted, notebook_velocity, np.nan), axis=1
    )
    notebook_finite = notebook_frame[np.isfinite(notebook_frame)]
    notebook_center = (
        float(np.nanmedian(notebook_finite)) if notebook_finite.size else np.nan
    )
    notebook_scatter = robust_scatter(notebook_finite)
    notebook_order_values = notebook_velocity[notebook_accepted]
    notebook_near_zero = (
        float(np.mean(np.abs(notebook_order_values) <= 0.50))
        if notebook_order_values.size else np.nan
    )
    summary = {
        "dataset": label,
        "frames": series.n_frames,
        "orders": series.n_orders,
        "telluric_rich_orders": [int(x) for x in np.asarray(series.orders)[rich_t]],
        "oh_rich_orders": [int(x) for x in np.asarray(series.orders)[rich_o]],
        "both_rich_orders": [int(x) for x in np.asarray(series.orders)[rich_t & rich_o]],
        "paired_accepted_cells": int(delta.size),
        "median_paired_centered_oh_minus_telluric_ms": (
            float(np.nanmedian(delta)) if delta.size else None),
        "robust_scatter_oh_minus_telluric_ms": (float(robust_scatter(delta)) if delta.size else None),
        "rms_oh_minus_telluric_ms": (float(np.sqrt(np.nanmean(delta**2))) if delta.size else None),
        "per_order_absolute_oh_minus_telluric_ms": offsets,
        "source_counts": {name: int(np.count_nonzero(run.solution.source == name))
                          for name in ("telluric", "OH", "interpolated", "unavailable")},
        "telluric_postcheck": {
            "direct_anchor_cells_ccf": int(post.size),
            "ccf_median_residual_ms": float(np.nanmedian(post) * 1e3) if post.size else None,
            "ccf_rms_residual_ms": (
                float(np.sqrt(np.nanmean(post ** 2)) * 1e3) if post.size else None
            ),
            "ccf_robust_scatter_residual_ms": (
                float(robust_scatter(post) * 1e3) if post.size else None
            ),
            "pre_frame_common_drift_rms_ms": float(np.nanstd(pre_common) * 1e3),
            "post_ccf_frame_common_drift_rms_ms": float(np.nanstd(post_common) * 1e3),
            "direct_anchor_cells_refit": int(post_refit.size),
            "refit_median_residual_ms": (
                float(np.nanmedian(post_refit) * 1e3) if post_refit.size else None
            ),
            "refit_rms_residual_ms": (
                float(np.sqrt(np.nanmean(post_refit ** 2)) * 1e3)
                if post_refit.size else None
            ),
            "refit_robust_scatter_residual_ms": (
                float(robust_scatter(post_refit) * 1e3) if post_refit.size else None
            ),
            "post_refit_frame_common_drift_rms_ms": (
                float(np.nanstd(post_refit_common) * 1e3)
            ),
            "ccf_method": (
                "physical zero-shift telluric template, fixed interior Pearson CCF, "
                "re-measured after WCS application"
            ),
            "refit_method": (
                "same physical telluric amplitude/continuum/shift estimator used "
                "for the direct calibration anchors, rerun after WCS application"
            ),
            "notebook_equivalent": {
                "zero_tolerance_kms": 0.50,
                "frame_median_residual_kms": notebook_center,
                "frame_robust_scatter_kms": notebook_scatter,
                "frame_peak_to_peak_kms": (
                    float(np.ptp(notebook_finite)) if notebook_finite.size else None
                ),
                "exposure_order_fraction_within_tolerance": notebook_near_zero,
                "residuals_essentially_zero": bool(
                    np.isfinite(notebook_center)
                    and np.isfinite(notebook_scatter)
                    and abs(notebook_center) <= 0.50
                    and notebook_scatter <= 0.50
                ),
                "method": (
                    "per-exposure fitted physical telluric template evaluated at zero "
                    "shift on the corrected common grid; fixed-support Pearson CCF"
                ),
            },
        },
        "solution": str(solution_path.resolve()),
        "telluric_residual_csv": str(residual_csv.resolve()),
        "pdf": str(output.resolve()),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output.resolve()}", flush=True)
    print(f"Wrote {solution_path.resolve()}", flush=True)
    print(f"Wrote {residual_csv.resolve()}", flush=True)
    print(f"Wrote {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
