#!/usr/bin/env python3
"""Compare WARP-only and final hybrid-wavecal Decanter spectra with SERVAL.

The packaging and two-chronological-bin statistics follow the local
``TRAPPIST1_wavecal_exojax_standalone.ipynb``.  The comparison uses identical
exposures, pixels, masks, and SERVAL settings; only the WAVE extension differs.
Out-of-transit exposures are the default so an RM/transit signal is not counted
as wavelength instability.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import median_filter

from decanter.wavecal.measure import continuum_normalize, highpass, robust_scatter
from decanter.wavecal.series import load_series
from decanter.wavecal.solution import C_KMS, WavecalSolution

WORKSPACE = Path(
    os.environ.get("DECANTER_VALIDATION_ROOT", Path(__file__).resolve().parents[2])
)
REPO = WORKSPACE / "decanter"
REDUCTIONS = WORKSPACE / "outputs/decanter_reductions"
SOLUTIONS = REPO / "output/pdf"
OUTDIR = WORKSPACE / "outputs/serval_hybrid_validation"
SERVAL_ROOT = Path(
    os.environ.get("SERVAL", str(Path.home() / "mzechmeister/serval"))
).resolve()
SERVAL_SRC = SERVAL_ROOT / "src"
RUNTIME_SRC = OUTDIR / "serval_runtime/src"
LCO = EarthLocation.from_geodetic(
    lon=-70.6915 * u.deg, lat=-29.0154 * u.deg, height=2380.0 * u.m
)
EDGE = 64
OVERSIZE = 100
NOMINAL_R = {"WIDE": 28_000.0, "HIRES-Y": 68_000.0, "HIRES-J": 68_000.0}
EPHEMERIS = {
    "wasp69b": dict(period_days=3.8681382, t0=2455748.83344, t14_days=0.0929),
    "toi2109b": dict(period_days=0.67247414, t0=2459378.459370,
                     t14_days=1.801 / 24.0),
    "toi3486b": dict(period_days=2.21778105, t0=2459797.24218,
                     t14_days=1.717 / 24.0),
}


def orbital_phase(bjd, dataset):
    ephem = EPHEMERIS[dataset]
    period = ephem["period_days"]
    return ((bjd - ephem["t0"] + 0.5 * period) % period) / period - 0.5


def frame_times(series):
    times, bjd, berv, airmass, coords = [], [], [], [], []
    for meta in series.meta:
        start = Time(f"{meta['DATE-OBS']}T{meta['UT-STR']}", scale="utc")
        end = Time(f"{meta['DATE-OBS']}T{meta['UT-END']}", scale="utc")
        if end < start:
            end += 1.0 * u.day
        mid = start + 0.5 * (end - start)
        coordinate = SkyCoord(meta["RA"], meta["DEC"], unit=(u.hourangle, u.deg))
        light = mid.light_travel_time(coordinate, location=LCO)
        times.append(mid)
        bjd.append(float((mid.tdb + light).jd))
        berv.append(float(coordinate.radial_velocity_correction(
            obstime=mid, location=LCO).to_value(u.km / u.s)))
        airmass.append(float(meta.get("AIRMASS", np.nan) or np.nan))
        coords.append((str(meta["RA"]), str(meta["DEC"])))
    return times, np.asarray(bjd), np.asarray(berv), np.asarray(airmass), coords


def dilate(mask, radius=3):
    kernel = np.ones(2 * radius + 1, dtype=int)
    out = np.zeros_like(mask, dtype=bool)
    for j in range(mask.shape[1]):
        out[:, j] = np.convolve(mask[:, j].astype(int), kernel, mode="same") > 0
    return out


def spectral_mask(series, solution):
    """Common atmospheric/sky mask, identical for both wave solutions."""
    width = max(51, int(round(151 * 0.96 / np.median(series.dv_pix_kms))) | 1)
    normalized = np.empty_like(series.obj, dtype=float)
    for i in range(series.n_frames):
        for j in range(series.n_orders):
            normalized[i, :, j] = continuum_normalize(series.obj[i, :, j], width)
    median_object = np.nanmedian(normalized, axis=0)
    rich_orders = set(int(order) for order in solution.meta.get("telluric_rich", []))
    atmospheric = np.zeros_like(median_object, dtype=bool)
    for j, order in enumerate(series.orders):
        if order in rich_orders:
            atmospheric[:, j] = median_object[:, j] < 0.90
    atmospheric = dilate(atmospheric, radius=3)

    sky_mask = np.zeros_like(atmospheric)
    if series.sky is not None:
        median_sky = np.nanmedian(series.sky, axis=0)
        for j in range(series.n_orders):
            feature = highpass(median_sky[:, j], 101)
            interior = feature[EDGE:-EDGE]
            sigma = robust_scatter(interior)
            if np.isfinite(sigma) and sigma > 0:
                sky_mask[:, j] = feature > 5.0 * sigma
        sky_mask = dilate(sky_mask, radius=3)
    return atmospheric, sky_mask


def aligned_velocity(solution, series):
    rows = [solution.frame_index(frame_id) for frame_id in series.frame_ids]
    columns = [solution.order_index(order) for order in series.orders]
    return solution.velocity[np.ix_(rows, columns)]


def prepare_runtime(n_orders, n_pixels, resolution):
    if not RUNTIME_SRC.exists():
        shutil.copytree(SERVAL_SRC, RUNTIME_SRC)
    pmin, pmax = EDGE + OVERSIZE, n_pixels - EDGE - OVERSIZE
    if pmax <= pmin:
        raise ValueError("spectra are too short for the SERVAL template margins")
    oset = f"0:{n_orders}"
    adapter = f'''from __future__ import print_function
from read_spec import *
name = "WINERED"
obsname = None
obsloc = {{"lat": -29.0154, "lon": -70.6915, "elevation": 2380.0}}
R = {float(resolution)!r}
iomax = {int(n_orders)}
pmin = {int(pmin)}
pmax = {int(pmax)}
oset = "{oset}"
coset = "{oset}"
pat = "*.fits"
maskfile = None
skyfile = None
atmspec = None
snmax = 1.0e9
def scan(self, s, pfits=True):
    self.hdulist = pyfits.open(s, memmap=False)
    self.header = hdr = self.hdulist[0].header
    self.instname = name
    self.drsberv = float(hdr["BERV"])
    self.drsbjd = float(hdr["BJD"])
    self.dateobs = hdr["DATE-OBS"]
    self.mjd = float(hdr["MJD-OBS"])
    self.drift = 0.0
    self.e_drift = 0.0
    self.sn55 = float(hdr.get("SNRREF", 20.0))
    self.fileid = hdr.get("TIMEID", s)
    self.timeid = self.fileid
    self.calmode = ""
    self.ccf.rvc = 0.0
    self.ccf.err_rvc = np.nan
    self.ra = hdr.get("RA", "00:00:00")
    self.de = hdr.get("DEC", "00:00:00")
    self.airmass = float(hdr.get("AIRMASS", np.nan))
    self.exptime = float(hdr.get("EXPTIME", 0.0))
    self.tmmean = 0.5
def data(self, orders=np.s_[:], pfits=True):
    h = self.hdulist
    f = np.asarray(h["SPEC"].data[orders], dtype=float)
    w = np.asarray(h["WAVE"].data[orders], dtype=float)
    e = np.asarray(h["SIG"].data[orders], dtype=float)
    bpmap = np.asarray(h["BPMAP"].data[orders], dtype=int)
    bad = ~np.isfinite(f) | ~np.isfinite(w) | ~np.isfinite(e) | (e <= 0)
    bpmap[bad] |= flag.nan
    with np.errstate(invalid="ignore"):
        bpmap[f < -3.0 * e] |= flag.neg
    return w, f, e, bpmap
'''
    (RUNTIME_SRC / "inst_WINERED.py").write_text(adapter)
    return oset


def package(series, velocity, keep, dataset, method, timing, masks):
    times, bjd, berv, airmass, coords = timing
    atmospheric, sky_mask = masks
    indices = np.where(keep)[0]
    root = OUTDIR / dataset / "input" / method
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    manifest = []
    for output_index, i in enumerate(indices):
        flux = np.asarray(series.obj[i], dtype=float).T
        wave = np.asarray(series.wave, dtype=float).T.copy()
        if velocity is not None:
            wave /= 1.0 + velocity[i, :, None] / C_KMS
        fraction = np.clip(np.nan_to_num(series.noise_fraction[i], nan=0.02),
                           2.0e-4, 0.30)
        level = np.abs(np.nanmedian(flux, axis=1))
        error = np.repeat((level * fraction)[:, None], series.n_pixels, axis=1)
        bpmap = np.zeros_like(flux, dtype=np.int32)
        bpmap[:, :EDGE] |= 1
        bpmap[:, -EDGE:] |= 1
        bpmap[atmospheric.T] |= 8
        bpmap[sky_mask.T] |= 16
        bad = ~np.isfinite(flux) | ~np.isfinite(wave) | ~np.isfinite(error) | (error <= 0)
        bpmap[bad] |= 1
        good = (bpmap == 0) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
        snr = float(np.nanmedian(np.abs(flux[good]) / error[good]))
        header = fits.Header()
        header["OBJECT"] = str(series.meta[i].get("OBJECT", dataset))
        header["INSTRUME"] = "WINERED"
        header["DATE-OBS"] = times[i].utc.isot
        header["MJD-OBS"] = float(times[i].utc.mjd)
        header["BJD"] = float(bjd[i])
        header["BERV"] = float(berv[i])
        header["RA"], header["DEC"] = coords[i]
        header["AIRMASS"] = float(airmass[i]) if np.isfinite(airmass[i]) else 1.0
        header["EXPTIME"] = float(series.meta[i].get("EXPTIME", 0.0) or 0.0)
        header["SNRREF"] = snr
        header["TIMEID"] = series.frame_ids[i]
        header["WCMETH"] = method
        path = root / f"exp_{output_index:04d}.fits"
        fits.HDUList([
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(flux.astype(np.float32), name="SPEC"),
            fits.ImageHDU(wave, name="WAVE"),
            fits.ImageHDU(error.astype(np.float32), name="SIG"),
            fits.ImageHDU(bpmap, name="BPMAP"),
            fits.ImageHDU(np.asarray(series.orders, dtype=np.int32), name="ORDER"),
        ]).writeto(path, overwrite=True)
        manifest.append(dict(local_index=output_index, series_index=int(i),
                             frame_id=series.frame_ids[i], BJD_TDB=float(bjd[i]),
                             path=str(path)))
    return root, pd.DataFrame(manifest)


def run_serval(input_dir, manifest, dataset, method, oset):
    run_dir = OUTDIR / dataset / "runs" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    target = f"{dataset}_{method}"
    target_dir = run_dir / target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    command = [
        sys.executable, str(RUNTIME_SRC / "serval.py"), target,
        str(input_dir.resolve()), "-inst", "WINERED", "-brvref", "DRS",
        "-targrv", "0", "-tplrv", "0", "-oset", oset, "-coset", oset,
        "-snmin", "0", "-snmax", "1000000000", "-niter", "2",
        "-safemode", "2", "-cache",
    ]
    env = os.environ.copy()
    companion = SERVAL_ROOT.parent / "python"
    env["PYTHONPATH"] = str(companion) + os.pathsep + env.get("PYTHONPATH", "")
    # The project conda environment carries gnuplot even when it was not
    # activated by the parent shell.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(command, cwd=run_dir, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    log = run_dir / "serval.log"
    log.write_text(result.stdout)
    rvo_path = target_dir / f"{target}.rvo.dat"
    if result.returncode != 0 or not rvo_path.exists():
        raise RuntimeError(
            f"SERVAL failed for {dataset}/{method} (rc={result.returncode})\n"
            + "\n".join(result.stdout.splitlines()[-50:])
        )
    rvo = np.genfromtxt(rvo_path)
    if rvo.ndim == 1:
        rvo = rvo[None, :]
    rows = []
    for output in rvo:
        nearest = int(np.nanargmin(np.abs(manifest.BJD_TDB.to_numpy() - output[0])))
        source = manifest.iloc[nearest]
        rows.append(dict(dataset=dataset, method=method,
                         series_index=int(source.series_index), frame_id=source.frame_id,
                         BJD_TDB=float(output[0]), RV_mps=float(output[1]),
                         E_RV_mps=float(output[2])))
    return pd.DataFrame(rows), str(rvo_path), str(log)


def summarize(table, n_orders, orders):
    rv = table.RV_mps.to_numpy(float)
    centered = rv - np.nanmedian(rv)
    table = table.copy()
    table["RV_centered_mps"] = centered
    ordered = table.sort_values("BJD_TDB")
    bins = np.array_split(np.arange(len(ordered)), 2)
    bin_values, bin_errors = [], []
    for indices in bins:
        local = ordered.iloc[indices]
        values = local.RV_centered_mps.to_numpy(float)
        errors = local.E_RV_mps.to_numpy(float)
        good = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        weight = 1.0 / errors[good] ** 2
        bin_values.append(float(np.sum(weight * values[good]) / np.sum(weight)))
        bin_errors.append(float(np.sqrt(1.0 / np.sum(weight))))
    bins_centered = np.asarray(bin_values) - np.mean(bin_values)
    summary = dict(
        n_orders=n_orders, orders=",".join(str(x) for x in orders),
        n_exposures=int(np.isfinite(centered).sum()),
        exposure_rms_mps=float(np.nanstd(centered)),
        exposure_mad_mps=float(robust_scatter(centered)),
        median_exposure_error_mps=float(np.nanmedian(table.E_RV_mps)),
        two_bin_separation_mps=float(abs(bin_values[1] - bin_values[0])),
        two_bin_rms_mps=float(np.std(bins_centered)),
        median_two_bin_error_mps=float(np.median(bin_errors)),
    )
    return table, summary


def validate_dataset(dataset, use_all=False):
    series = load_series(REDUCTIONS / dataset)
    solution = WavecalSolution.load_npz(SOLUTIONS / f"{dataset}_wavecal_solution.npz")
    velocity = aligned_velocity(solution, series)
    timing = frame_times(series)
    bjd = timing[1]
    phase = orbital_phase(bjd, dataset)
    half = 0.5 * EPHEMERIS[dataset]["t14_days"] / EPHEMERIS[dataset]["period_days"]
    keep = np.ones(series.n_frames, dtype=bool) if use_all else np.abs(phase) > half
    masks = spectral_mask(series, solution)
    oset = prepare_runtime(series.n_orders, series.n_pixels,
                            NOMINAL_R[series.instmode])
    raw_tables, run_paths = {}, {}
    for method, candidate in (("warp_only", None), ("hybrid", velocity)):
        print(f"{dataset}: packaging/running {method} ({keep.sum()} exposures)", flush=True)
        input_dir, manifest = package(
            series, candidate, keep, dataset, method, timing, masks
        )
        table, rvo_path, log_path = run_serval(
            input_dir, manifest, dataset, method, oset
        )
        raw_tables[method] = table
        run_paths[method] = (rvo_path, log_path)
    common_indices = set(raw_tables["warp_only"].series_index).intersection(
        set(raw_tables["hybrid"].series_index)
    )
    if len(common_indices) < 4:
        raise RuntimeError(f"too few common SERVAL exposures for {dataset}")

    exposure_tables, summaries = [], []
    for method in ("warp_only", "hybrid"):
        table = raw_tables[method][
            raw_tables[method].series_index.isin(common_indices)
        ].copy()
        table, summary = summarize(table, series.n_orders, series.orders)
        rvo_path, log_path = run_paths[method]
        summary.update(dataset=dataset, method=method, selection=("all" if use_all else "OOT"),
                       rvo_file=rvo_path, log_file=log_path)
        exposure_tables.append(table)
        summaries.append(summary)
        print(f"  {method} on {len(common_indices)} common exposures: "
              f"RMS={summary['exposure_rms_mps']:.1f}, "
              f"MAD={summary['exposure_mad_mps']:.1f}, "
              f"2-bin separation={summary['two_bin_separation_mps']:.1f} m/s",
              flush=True)
    return pd.concat(exposure_tables, ignore_index=True), pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=tuple(EPHEMERIS),
                        default=list(EPHEMERIS))
    parser.add_argument("--all-exposures", action="store_true")
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    exposures, summaries = [], []
    for dataset in args.datasets:
        exposure, summary = validate_dataset(dataset, args.all_exposures)
        exposures.append(exposure)
        summaries.append(summary)
    exposure_table = pd.concat(exposures, ignore_index=True)
    summary_table = pd.concat(summaries, ignore_index=True)
    exposure_table.to_csv(OUTDIR / "serval_exposure_rvs.csv", index=False)
    summary_table.to_csv(OUTDIR / "serval_stability_summary.csv", index=False)

    fig, axes = plt.subplots(len(args.datasets), 1, figsize=(10, 3.3 * len(args.datasets)),
                             constrained_layout=True, squeeze=False)
    for axis, dataset in zip(axes[:, 0], args.datasets):
        local = exposure_table[exposure_table.dataset == dataset]
        t0 = local.BJD_TDB.min()
        for method, color in (("warp_only", "0.5"), ("hybrid", "tab:blue")):
            rows = local[local.method == method]
            axis.errorbar((rows.BJD_TDB - t0) * 24.0, rows.RV_centered_mps,
                          yerr=rows.E_RV_mps, fmt=".", ms=4, lw=0.6,
                          color=color, alpha=0.8, label=method)
        axis.axhline(0, color="k", lw=0.7)
        axis.set(title=f"{dataset}: SERVAL OOT relative RV", xlabel="Hours",
                 ylabel="Median-centered RV (m/s)")
        axis.legend(frameon=False)
    fig.savefig(OUTDIR / "serval_rv_comparison.png", dpi=200)
    plt.close(fig)
    print(summary_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
