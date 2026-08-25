"""Optional post-wavecal SERVAL stellar-RV stability diagnostic.

The physical wavelength calibration is already encoded in each reduction's
WCS before this module is called. SERVAL reads those calibrated wavelengths
as given and serves only as the stellar RV estimator.
"""

from __future__ import annotations

import csv
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from decanter.wavecal.config import NOMINAL_RESOLVING_POWER
from decanter.wavecal.series import Series, from_reductions, load_series
from decanter.wavecal.solution import WavecalSolution

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

LCO = EarthLocation.from_geodetic(
    lon=-70.6915 * u.deg, lat=-29.0154 * u.deg, height=2380.0 * u.m
)
EDGE_PIXELS = 64
TEMPLATE_OVERSIZE_PIXELS = 100
SERVAL_FLAG_NAN = 1
SERVAL_FLAG_ATM = 8
SERVAL_FLAG_SKY = 16


@dataclass(frozen=True)
class RVStabilityResult:
    """Science products from one post-wavecal SERVAL run."""

    target: str
    wavecal_mode: str
    orders: tuple[int, ...]
    time_jd_utc: np.ndarray
    bjd_tdb: np.ndarray
    berv_kms: np.ndarray
    rv_mps: np.ndarray
    rv_error_mps: np.ndarray
    residual_mps: np.ndarray
    bin_time_jd_utc: np.ndarray
    bin_residual_mps: np.ndarray
    bin_error_mps: np.ndarray
    exposure_rms_mps: float
    exposure_mad_mps: float
    two_bin_rms_mps: float
    two_bin_mad_mps: float
    excluded_in_transit: int
    transit_window_time_jd_utc: np.ndarray
    product_path: Path
    table_path: Path
    figure_pdf: Path
    figure_png: Path
    log_path: Path


@dataclass(frozen=True, slots=True)
class TransitEphemeris:
    """Ephemeris used to exclude the transit from the stability test."""

    period_days: float
    transit_midpoint_bjd_tdb: float
    transit_duration_hours: float

    def __post_init__(self) -> None:
        if self.period_days <= 0:
            raise ValueError("transit period_days must be positive")
        if self.transit_midpoint_bjd_tdb <= 0:
            raise ValueError("transit_midpoint_bjd_tdb must be positive")
        if self.transit_duration_hours <= 0:
            raise ValueError("transit_duration_hours must be positive")

    @classmethod
    def from_toml(cls, path: str | Path) -> "TransitEphemeris":
        """Read the ``[system]`` ephemeris from a Decanter HRCCS config."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as stream:
            payload: dict[str, Any] = tomllib.load(stream)
        system = payload.get("system", payload)
        try:
            return cls(
                period_days=float(system["period_days"]),
                transit_midpoint_bjd_tdb=float(system["transit_midpoint_bjd_tdb"]),
                transit_duration_hours=float(system["transit_duration_hours"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"{config_path} is missing transit ephemeris field {exc.args[0]!r}"
            ) from exc


def robust_scatter(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    center = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - center)))


def _dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    return np.convolve(mask.astype(int), np.ones(2 * radius + 1, dtype=int),
                       mode="same") > 0


def _telluric_mask(product: str | Path | None, series: Series,
                   threshold: float) -> np.ndarray:
    """Map the minimum fitted transmission onto SERVAL's common order grids."""
    out = np.zeros((series.n_orders, series.n_pixels), dtype=bool)
    if product is None or not Path(product).is_file():
        warnings.warn(
            "telluric_transmission.npz is absent; SERVAL will not mask telluric pixels",
            RuntimeWarning,
            stacklevel=2,
        )
        return out
    with np.load(Path(product), allow_pickle=False) as data:
        product_orders = np.asarray(data["orders"], dtype=int)
        wavelength = np.asarray(data["wavelength_angstrom"], dtype=float)
        transmission = np.asarray(data["transmission"], dtype=float)
    for j, order in enumerate(series.orders):
        matches = np.where(product_orders == order)[0]
        if not matches.size:
            continue
        source_order = int(matches[0])
        minimum = np.full(series.n_pixels, np.inf)
        available = np.zeros(series.n_pixels, dtype=bool)
        for exposure in range(transmission.shape[0]):
            local_wave = wavelength[exposure, source_order]
            local_transmission = transmission[exposure, source_order]
            finite = np.isfinite(local_wave) & np.isfinite(local_transmission)
            if np.count_nonzero(finite) < 2:
                continue
            order_index = np.argsort(local_wave[finite])
            mapped = np.interp(
                series.wave[:, j], local_wave[finite][order_index],
                local_transmission[finite][order_index], left=np.nan, right=np.nan,
            )
            good = np.isfinite(mapped)
            minimum[good] = np.minimum(minimum[good], mapped[good])
            available |= good
        out[j] = _dilate(available & (minimum < threshold))
    return out


def _oh_mask(wavecal_run, series: Series) -> np.ndarray:
    """Map fitted OH-line support from the wavecal run onto SERVAL's grid."""
    out = np.zeros((series.n_orders, series.n_pixels), dtype=bool)
    if wavecal_run is None or wavecal_run.oh_model is None:
        return out
    source_series = wavecal_run.series
    support = np.asarray(wavecal_run.oh_model.support, dtype=bool)
    for j, order in enumerate(series.orders):
        if order not in source_series.orders:
            continue
        source_order = source_series.order_index(order)
        mapped = np.interp(
            series.wave[:, j], source_series.wave[:, source_order],
            support[:, source_order].astype(float), left=0.0, right=0.0,
        ) > 0.25
        out[j] = _dilate(mapped)
    return out


def _frame_coordinates(series: Series):
    times, bjd, berv = [], [], []
    coordinates = []
    for time_jd, meta in zip(series.time_jd, series.meta, strict=True):
        if not np.isfinite(time_jd):
            raise ValueError("SERVAL RV stability requires finite exposure times")
        mid = Time(float(time_jd), format="jd", scale="utc")
        ra, dec = meta.get("RA"), meta.get("DEC")
        if ra is None or dec is None:
            raise ValueError("SERVAL RV stability requires RA and DEC in every spectrum")
        coordinate = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            light_time = mid.light_travel_time(coordinate, location=LCO)
            correction = coordinate.radial_velocity_correction(
                obstime=mid, location=LCO
            ).to_value(u.km / u.s)
        times.append(mid)
        bjd.append(float((mid.tdb + light_time).jd))
        berv.append(float(correction))
        coordinates.append(coordinate)
    return times, np.asarray(bjd), np.asarray(berv), coordinates


def _adapter_text(n_orders: int, n_pixels: int, resolution: float) -> tuple[str, str]:
    pmin = EDGE_PIXELS + TEMPLATE_OVERSIZE_PIXELS
    pmax = n_pixels - EDGE_PIXELS - TEMPLATE_OVERSIZE_PIXELS
    if pmax <= pmin:
        raise ValueError(
            "WINERED orders are too short for SERVAL's 100-pixel template oversizing"
        )
    order_set = f"0:{n_orders}"
    text = f'''from __future__ import print_function

from read_spec import *

name = "WINERED"
obsname = None
obsloc = {{"lat": -29.0154, "lon": -70.6915, "elevation": 2380.0}}
R = {float(resolution)!r}
iomax = {int(n_orders)}
pmin = {int(pmin)}
pmax = {int(pmax)}
oset = "{order_set}"
coset = "{order_set}"
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
    f = np.asarray(self.hdulist["SPEC"].data[orders], dtype=float)
    w = np.asarray(self.hdulist["WAVE"].data[orders], dtype=float)
    e = np.asarray(self.hdulist["SIG"].data[orders], dtype=float)
    bpmap = np.asarray(self.hdulist["BPMAP"].data[orders], dtype=int)
    bad = ~np.isfinite(f) | ~np.isfinite(w) | ~np.isfinite(e) | (e <= 0)
    bpmap[bad] |= flag.nan
    with np.errstate(invalid="ignore"):
        bpmap[f < -3.0 * e] |= flag.neg
    return w, f, e, bpmap
'''
    return text, order_set


def _resolve_serval_root(value: str | Path | None) -> Path:
    candidates = []
    if value is not None:
        candidates.append(Path(value))
    if os.environ.get("SERVAL"):
        candidates.append(Path(os.environ["SERVAL"]))
    candidates.extend((Path.home() / "mzechmeister" / "serval", Path.home() / "serval"))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "src" / "serval.py").is_file():
            return root
    searched = "\n".join(f"    {item.expanduser()}" for item in candidates)
    raise FileNotFoundError(
        "SERVAL was not found. Set --serval-dir or the SERVAL environment variable.\n"
        f"Searched:\n{searched}"
    )


def _prepare_runtime(source_root: Path, temporary_root: Path) -> Path:
    """Copy SERVAL source so its instrument adapter is never modified in place."""
    runtime = temporary_root / "serval_runtime"
    shutil.copytree(
        source_root / "src", runtime / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if (source_root / "lib").exists():
        (runtime / "lib").symlink_to(source_root / "lib", target_is_directory=True)
    return runtime


def _gnuplot_environment(environment: dict[str, str], temporary_root: Path) -> None:
    """Supply a no-op plotting process when SERVAL's optional GUI is unavailable."""
    if shutil.which("gnuplot", path=environment.get("PATH")) is not None:
        return
    binary_dir = temporary_root / "bin"
    binary_dir.mkdir()
    shim = binary_dir / "gnuplot"
    shim.write_text(
        '#!/bin/sh\nif [ "$1" = "-V" ]; then echo "gnuplot 5.4"; exit 0; fi\n'
        'while IFS= read -r line; do :; done\n'
    )
    shim.chmod(0o755)
    environment["PATH"] = str(binary_dir) + os.pathsep + environment.get("PATH", "")


def _write_inputs(series: Series, directory: Path, bpmap: np.ndarray,
                  times, bjd, berv, coordinates,
                  wave_by_frame: np.ndarray | None = None) -> None:
    if wave_by_frame is not None:
        wave_by_frame = np.asarray(wave_by_frame, dtype=float)
        expected = (series.n_frames, series.n_pixels, series.n_orders)
        if wave_by_frame.shape != expected:
            raise ValueError(
                f"wave_by_frame has shape {wave_by_frame.shape}, expected {expected}"
            )
    directory.mkdir(parents=True)
    for i, frame_id in enumerate(series.frame_ids):
        flux = np.asarray(series.obj[i], dtype=np.float64).T
        wave = np.asarray(
            series.wave if wave_by_frame is None else wave_by_frame[i],
            dtype=np.float64,
        ).T
        fraction = np.clip(np.nan_to_num(series.noise_fraction[i], nan=0.02), 2e-4, 0.3)
        level = np.abs(np.nanmedian(flux, axis=1))[:, None]
        error = np.repeat(level * fraction[:, None], series.n_pixels, axis=1)
        error = np.where(np.isfinite(error) & (error > 0), error, np.nan)
        local_bpmap = np.asarray(bpmap, dtype=np.int32).copy()
        local_bpmap[~np.isfinite(flux) | ~np.isfinite(wave) | ~np.isfinite(error)] |= (
            SERVAL_FLAG_NAN
        )
        good = np.isfinite(flux) & np.isfinite(error) & (error > 0) & (local_bpmap == 0)
        snr = float(np.nanmedian(np.abs(flux[good]) / error[good])) if np.any(good) else 20.0
        meta = series.meta[i]
        header = fits.Header()
        header["OBJECT"] = str(meta.get("OBJECT", "TARGET"))
        header["INSTRUME"] = "WINERED"
        header["DATE-OBS"] = times[i].utc.isot
        header["MJD-OBS"] = float(times[i].utc.mjd)
        header["BJD"] = float(bjd[i])
        header["BERV"] = float(berv[i])
        header["RA"] = coordinates[i].ra.to_string(unit=u.hourangle, sep=":")
        header["DEC"] = coordinates[i].dec.to_string(unit=u.deg, sep=":", alwayssign=True)
        header["AIRMASS"] = float(series.airmass[i]) if np.isfinite(series.airmass[i]) else 1.0
        header["EXPTIME"] = float(meta.get("EXPTIME", 0.0) or 0.0)
        header["SNRREF"] = snr
        header["TIMEID"] = str(frame_id)
        fits.HDUList([
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(flux, name="SPEC"),
            fits.ImageHDU(wave, name="WAVE"),
            fits.ImageHDU(error, name="SIG"),
            fits.ImageHDU(local_bpmap, name="BPMAP"),
            fits.ImageHDU(np.asarray(series.orders, dtype=np.int32), name="ORDER"),
        ]).writeto(directory / f"exp_{i:04d}.fits", overwrite=True)


def _run_serval(series: Series, output: Path, serval_root: Path, bpmap: np.ndarray,
                times, bjd, berv, coordinates,
                wave_by_frame: np.ndarray | None = None) -> tuple[np.ndarray, Path]:
    run_dir = output / "serval_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = output / "serval.log"
    target = "decanter_rv_stability"
    old_target = run_dir / target
    if old_target.exists():
        shutil.rmtree(old_target)

    with tempfile.TemporaryDirectory(prefix="decanter_serval_") as temporary_name:
        temporary = Path(temporary_name)
        runtime = _prepare_runtime(serval_root, temporary)
        resolution = NOMINAL_RESOLVING_POWER.get(series.instmode, 68_000.0)
        adapter, order_set = _adapter_text(series.n_orders, series.n_pixels, resolution)
        (runtime / "src" / "inst_WINERED.py").write_text(adapter)
        input_dir = temporary / "input"
        temporary_run_dir = temporary / "run"
        temporary_run_dir.mkdir()
        print(f"SERVAL: packaging {series.n_frames} calibrated exposures", flush=True)
        _write_inputs(
            series, input_dir, bpmap, times, bjd, berv, coordinates,
            wave_by_frame=wave_by_frame,
        )
        command = [
            sys.executable, str(runtime / "src" / "serval.py"), target,
            str(input_dir), "-inst", "WINERED", "-brvref", "DRS",
            "-targrv", "0", "-tplrv", "0", "-oset", order_set,
            "-coset", order_set, "-snmin", "0", "-snmax", "1000000000",
            "-niter", "2", "-safemode", "2", "-cache",
        ]
        environment = os.environ.copy()
        companion = serval_root.parent / "python"
        if companion.exists():
            environment["PYTHONPATH"] = str(companion) + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH") else ""
            )
        environment["PYTHONUNBUFFERED"] = "1"
        _gnuplot_environment(environment, temporary)
        print(
            f"SERVAL: fitting {series.n_frames} exposures x {series.n_orders} orders",
            flush=True,
        )
        last_bucket = -1
        def report_line(line):
            nonlocal last_bucket
            match = re.match(r"\s*(\d+)\s*/\s*(\d+)\s+(?:scan\s+)?\S+", line)
            if match:
                current, total = map(int, match.groups())
                bucket = int(10 * current / max(total, 1))
                if bucket != last_bucket or current == total:
                    print(f"SERVAL progress: {current}/{total}", flush=True)
                    last_bucket = bucket
            elif any(token in line for token in (
                "SERVAL - SpEctrum", "start time:", "creating template",
                "Iteration ", "RVs from iteration",
            )):
                print(line.strip(), flush=True)

        with log_path.open("w") as log_stream:
            process = subprocess.Popen(
                command, cwd=temporary_run_dir, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            try:
                while True:
                    ready, _, _ = select.select([process.stdout], [], [], 0.5)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            log_stream.write(line)
                            log_stream.flush()
                            report_line(line)
                    if process.poll() is not None:
                        while True:
                            ready, _, _ = select.select([process.stdout], [], [], 0.0)
                            if not ready:
                                break
                            line = process.stdout.readline()
                            if not line:
                                break
                            log_stream.write(line)
                            report_line(line)
                        break
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            return_code = process.wait()
            log_stream.flush()
        temporary_target = temporary_run_dir / target
        temporary_rvo = temporary_target / f"{target}.rvo.dat"
        if return_code != 0 or not temporary_rvo.is_file():
            tail = "\n".join(log_path.read_text().splitlines()[-60:])
            raise RuntimeError(
                f"SERVAL failed with return code {return_code}.\n"
                f"Last log lines:\n{tail}\nFull log: {log_path}"
            )
        shutil.copytree(temporary_target, old_target)
        rvo_path = old_target / f"{target}.rvo.dat"
        table = np.genfromtxt(rvo_path, dtype=float)
    if table.ndim == 1:
        table = table[None, :]
    if table.ndim != 2 or table.shape[1] < 5:
        raise RuntimeError(f"unexpected SERVAL rvo.dat format: {table.shape}")
    return table, log_path


def _match_serval_rows(table: np.ndarray, bjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # SERVAL writes rvo.dat in scan order. Prefer that exact mapping when the
    # complete table is present; this remains unambiguous when two source FITS
    # headers legitimately carry the same timestamp.
    if table.shape[0] == bjd.size and np.all(
        np.isfinite(table[:, 0]) & (np.abs(table[:, 0] - bjd) <= 1.0e-4)
    ):
        indices = np.arange(bjd.size, dtype=int)
        return indices, indices.copy()
    row_for_exposure = np.full(bjd.size, -1, dtype=int)
    used = set()
    for row, output_bjd in enumerate(table[:, 0]):
        index = int(np.nanargmin(np.abs(bjd - output_bjd)))
        if abs(float(bjd[index] - output_bjd)) > 1.0e-4 or index in used:
            raise RuntimeError(f"could not uniquely match SERVAL BJD {output_bjd}")
        row_for_exposure[index] = row
        used.add(index)
    keep = np.where(row_for_exposure >= 0)[0]
    return keep, row_for_exposure[keep]


def _inverse_variance_mean(values, errors) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    good = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if np.any(good):
        weight = 1.0 / errors[good] ** 2
        return (float(np.sum(weight * values[good]) / np.sum(weight)),
                float(np.sqrt(1.0 / np.sum(weight))), int(np.count_nonzero(good)))
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan"), float("nan"), 0
    local = values[finite]
    error = np.std(local, ddof=1) / np.sqrt(local.size) if local.size >= 2 else np.nan
    return float(np.mean(local)), float(error), int(local.size)


def _two_bins(time_jd, residual, error, excluded_window=None):
    good = np.isfinite(time_jd) & np.isfinite(residual)
    order = np.argsort(np.asarray(time_jd)[good])
    time_jd = np.asarray(time_jd)[good][order]
    residual = np.asarray(residual)[good][order]
    error = np.asarray(error)[good][order]
    bin_time = np.full(2, np.nan)
    bin_value = np.full(2, np.nan)
    bin_error = np.full(2, np.nan)
    bin_count = np.zeros(2, dtype=int)
    if residual.size < 2:
        return bin_time, bin_value, bin_error, bin_count
    groups = None
    if excluded_window is not None:
        window = np.asarray(excluded_window, dtype=float)
        before = np.where(time_jd < window[0])[0]
        after = np.where(time_jd > window[1])[0]
        if before.size and after.size:
            groups = (before, after)
    if groups is None:
        groups = np.array_split(np.arange(residual.size), 2)
    for index, group in enumerate(groups):
        bin_time[index] = np.mean(time_jd[group])
        bin_value[index], bin_error[index], bin_count[index] = _inverse_variance_mean(
            residual[group], error[group]
        )
    finite = np.isfinite(bin_value)
    if np.count_nonzero(finite) >= 2:
        bin_value[finite] -= np.mean(bin_value[finite])
    return bin_time, bin_value, bin_error, bin_count


def _safe_target(series: Series) -> str:
    value = str(series.meta[0].get("OBJECT", "target")).strip() or "target"
    return re.sub(r"[^A-Za-z0-9_.+-]+", "-", value).strip("-") or "target"


def _transit_selection(
    bjd_tdb: np.ndarray,
    time_jd_utc: np.ndarray,
    ephemeris: TransitEphemeris,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the in-transit mask and the observed transit's UTC window."""
    bjd_tdb = np.asarray(bjd_tdb, dtype=float)
    time_jd_utc = np.asarray(time_jd_utc, dtype=float)
    phase_days = (
        (bjd_tdb - ephemeris.transit_midpoint_bjd_tdb + 0.5 * ephemeris.period_days)
        % ephemeris.period_days
        - 0.5 * ephemeris.period_days
    )
    half_duration_days = 0.5 * ephemeris.transit_duration_hours / 24.0
    in_transit = np.abs(phase_days) <= half_duration_days

    cycle = int(np.rint(
        (np.nanmedian(bjd_tdb) - ephemeris.transit_midpoint_bjd_tdb)
        / ephemeris.period_days
    ))
    observed_midpoint_bjd = (
        ephemeris.transit_midpoint_bjd_tdb + cycle * ephemeris.period_days
    )
    # The barycentric light-time offset changes negligibly across one night.
    # Its measured median maps the BJD transit window onto the UTC plot axis.
    bjd_minus_utc = float(np.nanmedian(bjd_tdb - time_jd_utc))
    midpoint_utc = observed_midpoint_bjd - bjd_minus_utc
    window_utc = midpoint_utc + np.asarray(
        [-half_duration_days, half_duration_days], dtype=float
    )
    return in_transit, window_utc


def _retain_frames(series: Series, keep: np.ndarray) -> Series:
    """Return a Series containing only selected chronological exposures."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (series.n_frames,):
        raise ValueError("frame selection has the wrong shape")
    return replace(
        series,
        frame_ids=tuple(np.asarray(series.frame_ids)[keep]),
        obj=series.obj[keep],
        sky=None if series.sky is None else series.sky[keep],
        noise_fraction=series.noise_fraction[keep],
        time_jd=series.time_jd[keep],
        sky_time_jd=series.sky_time_jd[keep],
        airmass=series.airmass[keep],
        meta=[meta for meta, selected in zip(series.meta, keep, strict=True) if selected],
    )


def _retain_usable_orders(
    series: Series, masked: np.ndarray
) -> tuple[Series, np.ndarray, np.ndarray]:
    """Drop orders that cannot support SERVAL after fixed-interior masking."""
    pmin = EDGE_PIXELS + TEMPLATE_OVERSIZE_PIXELS
    pmax = series.n_pixels - EDGE_PIXELS - TEMPLATE_OVERSIZE_PIXELS
    interior_size = pmax - pmin
    minimum = max(200, int(np.ceil(0.10 * interior_size)))
    finite_flux = np.mean(np.isfinite(series.obj), axis=0).T >= 0.8
    usable_pixels = np.sum(~masked[:, pmin:pmax] & finite_flux[:, pmin:pmax], axis=1)
    keep = usable_pixels >= minimum
    if np.count_nonzero(keep) < 2:
        raise ValueError(
            "fewer than two orders retain enough pixels for SERVAL after masking"
        )
    dropped = np.asarray(series.orders)[~keep]
    if dropped.size:
        print(
            "SERVAL: excluding insufficiently sampled physical orders "
            + ", ".join(map(str, dropped)),
            flush=True,
        )
    local = replace(
        series,
        orders=tuple(np.asarray(series.orders)[keep].astype(int)),
        wave=series.wave[:, keep],
        dv_pix_kms=series.dv_pix_kms[keep],
        obj=series.obj[:, :, keep],
        sky=None if series.sky is None else series.sky[:, :, keep],
        noise_fraction=series.noise_fraction[:, keep],
    )
    return local, masked[keep], keep


def _save_products(output: Path, target: str, mode: str, series: Series,
                   time_jd, bjd, berv, rv, error, residual,
                   bin_time, bin_value, bin_error, bin_count,
                   ephemeris: TransitEphemeris, excluded_in_transit: int,
                   transit_window_time_jd_utc: np.ndarray):
    exposure_rms = float(np.nanstd(residual))
    exposure_mad = robust_scatter(residual)
    two_bin_rms = float(np.nanstd(bin_value))
    two_bin_mad = robust_scatter(bin_value)
    product = output / "serval_rv_stability.npz"
    np.savez_compressed(
        product, target=np.asarray(target), wavecal_mode=np.asarray(mode),
        orders=np.asarray(series.orders, dtype=np.int32),
        time_jd_utc=time_jd, bjd_tdb=bjd, berv_kms=berv,
        rv_mps=rv, rv_error_mps=error, observation_centered_rv_mps=residual,
        bin_time_jd_utc=bin_time, bin_residual_mps=bin_value,
        bin_error_mps=bin_error, bin_n_exposures=bin_count,
        exposure_rms_mps=np.asarray(exposure_rms),
        exposure_mad_mps=np.asarray(exposure_mad),
        two_bin_rms_mps=np.asarray(two_bin_rms), two_bin_mad_mps=np.asarray(two_bin_mad),
        excluded_in_transit=np.asarray(excluded_in_transit, dtype=np.int32),
        transit_window_time_jd_utc=transit_window_time_jd_utc,
        period_days=np.asarray(ephemeris.period_days),
        transit_midpoint_bjd_tdb=np.asarray(ephemeris.transit_midpoint_bjd_tdb),
        transit_duration_hours=np.asarray(ephemeris.transit_duration_hours),
    )
    table_path = output / "serval_rv_stability.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "frame_id", "time_jd_utc", "bjd_tdb", "berv_kms", "rv_mps",
            "rv_error_mps", "observation_centered_rv_mps",
        ))
        writer.writeheader()
        for index, frame_id in enumerate(series.frame_ids):
            writer.writerow({
                "frame_id": frame_id, "time_jd_utc": time_jd[index],
                "bjd_tdb": bjd[index], "berv_kms": berv[index], "rv_mps": rv[index],
                "rv_error_mps": error[index],
                "observation_centered_rv_mps": residual[index],
            })
    summary = {
        "schema": "decanter.serval-rv-stability.v2", "target": target,
        "wavecal_mode": mode, "n_exposures": int(np.count_nonzero(np.isfinite(rv))),
        "orders": [int(order) for order in series.orders],
        "excluded_in_transit": int(excluded_in_transit),
        "period_days": ephemeris.period_days,
        "transit_midpoint_bjd_tdb": ephemeris.transit_midpoint_bjd_tdb,
        "transit_duration_hours": ephemeris.transit_duration_hours,
        "exposure_rms_mps": exposure_rms,
        "exposure_mad_mps": exposure_mad, "two_bin_rms_mps": two_bin_rms,
        "two_bin_mad_mps": two_bin_mad,
    }
    (output / "serval_rv_stability.json").write_text(json.dumps(summary, indent=2))
    return product, table_path, exposure_rms, exposure_mad, two_bin_rms, two_bin_mad


def _plot(output: Path, target: str, mode: str, time_jd, residual, error,
          bin_time, bin_value, bin_error, exposure_rms, exposure_mad,
          two_bin_rms, two_bin_mad, transit_window_time_jd_utc,
          excluded_in_transit) -> tuple[Path, Path]:
    matplotlib_cache = Path(tempfile.gettempdir()) / "decanter_matplotlib_cache"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    color = "#2ca02c"
    figure, axis = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    window = Time(
        np.asarray(transit_window_time_jd_utc), format="jd"
    ).to_datetime()
    axis.axvspan(
        window[0], window[1], color="0.88", alpha=0.75, lw=0,
        label=f"transit excluded ({excluded_in_transit} exposures)", zorder=0,
    )
    good = np.isfinite(time_jd) & np.isfinite(residual)
    axis.errorbar(
        Time(np.asarray(time_jd)[good], format="jd").to_datetime(), residual[good],
        yerr=error[good], fmt="D", linestyle="none", ms=4.7, color=color,
        alpha=0.40, elinewidth=0.5, capsize=1.2,
        label=(
            f"SERVAL OOT — all usable orders: exp RMS={exposure_rms:.1f}, "
            rf"MAD$\sigma$={exposure_mad:.1f}; pre/post RMS={two_bin_rms:.1f}, "
            rf"MAD$\sigma$={two_bin_mad:.1f} m s$^{{-1}}$"
        ), zorder=2,
    )
    bin_good = np.isfinite(bin_time) & np.isfinite(bin_value)
    axis.errorbar(
        Time(np.asarray(bin_time)[bin_good], format="jd").to_datetime(),
        bin_value[bin_good], yerr=bin_error[bin_good], fmt="D", linestyle="none",
        ms=11, color=color, markeredgecolor="k", markeredgewidth=1.2,
        capsize=3, elinewidth=1.0, zorder=5,
    )
    axis.axhline(0, color="0.25", lw=0.7)
    axis.set_xlabel("Time (UTC)")
    axis.set_ylabel("observation-centered stellar RV residual (m s$^{-1}$)")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.set_title(
        f"{target}: {mode.replace('_', ' ')} wavelength calibration — "
        "out-of-transit only"
    )
    axis.legend(frameon=False, fontsize=8, loc="best")
    pdf = output / "serval_rv_stability.pdf"
    png = output / "serval_rv_stability.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=220)
    plt.close(figure)
    return pdf, png


def run_serval_rv_stability(
    transit_series,
    output_root: str | Path,
    *,
    ephemeris: TransitEphemeris | str | Path,
    serval_root: str | Path | None = None,
    telluric_threshold: float = 0.995,
) -> RVStabilityResult:
    """Run SERVAL after physical wavecal on an in-memory TransitSeries."""
    if transit_series.wavecal_solution is None:
        raise ValueError("SERVAL RV stability requires a physical wavelength calibration")
    series = from_reductions(
        transit_series.reductions, fsr_cut=transit_series.wavecal_run.config.fsr_cut
        if transit_series.wavecal_run is not None else None,
    )
    return _run(
        series, output_root, transit_series.wavecal_solution.mode,
        serval_root=serval_root, telluric_threshold=telluric_threshold,
        wavecal_run=transit_series.wavecal_run, target_hint=None,
        ephemeris=_resolve_ephemeris(ephemeris),
    )


def run_serval_rv_stability_directory(
    reduction_root: str | Path,
    *,
    ephemeris: TransitEphemeris | str | Path,
    serval_root: str | Path | None = None,
    telluric_threshold: float = 0.995,
) -> RVStabilityResult:
    """Run the same diagnostic on an already-written Decanter reduction."""
    root = Path(reduction_root).expanduser().resolve()
    solution_path = root / "wavecal_solution.npz"
    if not solution_path.is_file():
        raise ValueError(f"SERVAL RV stability requires {solution_path}")
    solution = WavecalSolution.load_npz(solution_path)
    return _run(
        load_series(root), root, solution.mode, serval_root=serval_root,
        telluric_threshold=telluric_threshold, wavecal_run=None,
        target_hint=root.name, ephemeris=_resolve_ephemeris(ephemeris),
    )


def _resolve_ephemeris(
    value: TransitEphemeris | str | Path,
) -> TransitEphemeris:
    if isinstance(value, TransitEphemeris):
        return value
    return TransitEphemeris.from_toml(value)


def _run(series: Series, output_root: str | Path, wavecal_mode: str, *,
         serval_root: str | Path | None, telluric_threshold: float,
         wavecal_run, target_hint: str | None,
         ephemeris: TransitEphemeris) -> RVStabilityResult:
    if not 0.0 < telluric_threshold <= 1.0:
        raise ValueError("telluric_threshold must be in (0, 1]")
    if series.n_frames < 2:
        raise ValueError("SERVAL RV stability requires at least two exposures")
    output_root = Path(output_root).expanduser().resolve()
    output = output_root / "serval_rv_stability"
    output.mkdir(parents=True, exist_ok=True)
    all_times, all_bjd, all_berv, all_coordinates = _frame_coordinates(series)
    in_transit, transit_window_time_jd_utc = _transit_selection(
        all_bjd, series.time_jd, ephemeris
    )
    excluded_in_transit = int(np.count_nonzero(in_transit))
    keep_out_of_transit = ~in_transit
    if np.count_nonzero(keep_out_of_transit) < 4:
        raise ValueError(
            "SERVAL RV stability requires at least four out-of-transit exposures"
        )
    print(
        f"SERVAL: excluding {excluded_in_transit} in-transit exposures; "
        f"using {np.count_nonzero(keep_out_of_transit)} out of transit",
        flush=True,
    )
    series = _retain_frames(series, keep_out_of_transit)
    times = [time for time, keep in zip(all_times, keep_out_of_transit, strict=True) if keep]
    full_bjd = all_bjd[keep_out_of_transit]
    full_berv = all_berv[keep_out_of_transit]
    coordinates = [
        coordinate
        for coordinate, keep in zip(all_coordinates, keep_out_of_transit, strict=True)
        if keep
    ]
    product = output_root / "telluric_transmission.npz"
    telluric = _telluric_mask(product if product.is_file() else None, series,
                              telluric_threshold)
    sky = _oh_mask(wavecal_run, series)
    series, combined_mask, keep_orders = _retain_usable_orders(
        series, telluric | sky
    )
    telluric = telluric[keep_orders]
    sky = sky[keep_orders]
    bpmap = np.zeros((series.n_orders, series.n_pixels), dtype=np.int32)
    bpmap[:, :EDGE_PIXELS] |= SERVAL_FLAG_NAN
    bpmap[:, -EDGE_PIXELS:] |= SERVAL_FLAG_NAN
    bpmap[telluric] |= SERVAL_FLAG_ATM
    bpmap[sky] |= SERVAL_FLAG_SKY

    table, log_path = _run_serval(
        series, output, _resolve_serval_root(serval_root), bpmap,
        times, full_bjd, full_berv, coordinates,
    )
    exposure_indices, table_rows = _match_serval_rows(table, full_bjd)
    time_jd = np.asarray(series.time_jd, dtype=float)
    rv = np.full(series.n_frames, np.nan)
    error = np.full(series.n_frames, np.nan)
    bjd = full_bjd.copy()
    berv = full_berv.copy()
    rv[exposure_indices] = table[table_rows, 1]
    error[exposure_indices] = table[table_rows, 2]
    residual = rv - np.nanmedian(rv)
    bin_time, bin_value, bin_error, bin_count = _two_bins(
        time_jd, residual, error, transit_window_time_jd_utc
    )
    target = _safe_target(series)
    if target.upper() in {"UNKNOWN", "TARGET"} and target_hint:
        target = re.sub(r"[^A-Za-z0-9_.+-]+", "-", target_hint).strip("-")
    (product_path, table_path, exposure_rms, exposure_mad,
     two_bin_rms, two_bin_mad) = _save_products(
        output, target, wavecal_mode, series, time_jd, bjd, berv, rv, error, residual,
        bin_time, bin_value, bin_error, bin_count, ephemeris,
        excluded_in_transit, transit_window_time_jd_utc,
    )
    pdf, png = _plot(
        output, target, wavecal_mode, time_jd, residual, error,
        bin_time, bin_value, bin_error, exposure_rms, exposure_mad,
        two_bin_rms, two_bin_mad, transit_window_time_jd_utc,
        excluded_in_transit,
    )
    print(
        f"SERVAL RV stability: RMS={exposure_rms:.1f} m/s, "
        f"MADsigma={exposure_mad:.1f} m/s; PDF -> {pdf}", flush=True,
    )
    return RVStabilityResult(
        target, wavecal_mode, series.orders, time_jd, bjd, berv, rv, error, residual,
        bin_time, bin_value, bin_error, exposure_rms, exposure_mad,
        two_bin_rms, two_bin_mad, excluded_in_transit,
        transit_window_time_jd_utc, product_path, table_path, pdf, png, log_path,
    )
