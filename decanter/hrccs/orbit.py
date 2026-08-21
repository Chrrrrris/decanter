"""Target metadata, barycentric correction, and transit coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Orbit:
    target_name: str
    ra_deg: float
    dec_deg: float
    stellar_rv_kms: float
    bjd_tdb: np.ndarray
    phase: np.ndarray
    in_transit: np.ndarray
    transit_weight: np.ndarray
    berv_kms: np.ndarray


def _target_name(config, metadata) -> str:
    if config.planet_name:
        return config.planet_name
    for row in metadata:
        value = str(row.get("OBJECT", "")).strip()
        if value and value.upper() not in {"UNKNOWN", "N/A"}:
            return value
    raise ValueError("planet_name could not be inferred from FITS metadata")


def _simbad(name: str, cache_dir: Path) -> tuple[float, float, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "simbad_target.json"
    if cache.exists():
        values = json.loads(cache.read_text())
        if values.get("query") == name:
            return float(values["ra_deg"]), float(values["dec_deg"]), float(values["rv_kms"])
    from astroquery.simbad import Simbad

    query = Simbad()
    query.add_votable_fields("velocity")
    table = query.query_object(name)
    if table is None or len(table) == 0:
        raise RuntimeError(f"SIMBAD returned no match for {name!r}")
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    columns = {column.lower(): column for column in table.colnames}
    def column(*names):
        for candidate in names:
            if candidate.lower() in columns:
                return columns[candidate.lower()]
        raise RuntimeError(f"SIMBAD response lacks all of {names}; columns={table.colnames}")
    coord = SkyCoord(str(table[column("ra")][0]), str(table[column("dec")][0]),
                     unit=(u.hourangle, u.deg))
    rv = float(table[column("rvz_radvel", "rvz_radvel_value", "velocities")][0])
    values = {"query": name, "ra_deg": coord.ra.deg, "dec_deg": coord.dec.deg,
              "rv_kms": rv, "source": "SIMBAD rvz_radvel"}
    cache.write_text(json.dumps(values, indent=2, sort_keys=True))
    return coord.ra.deg, coord.dec.deg, rv


def build_orbit(system, time_jd_utc, metadata, cache_dir: str | Path) -> Orbit:
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = bool(system.iers_auto_download)
    iers.conf.auto_max_age = None
    name = _target_name(system, metadata)
    if system.ra_deg is None or system.dec_deg is None or system.stellar_rv_kms is None:
        sim_ra, sim_dec, sim_rv = _simbad(name, Path(cache_dir).expanduser())
    else:
        sim_ra = sim_dec = sim_rv = np.nan
    ra = sim_ra if system.ra_deg is None else system.ra_deg
    dec = sim_dec if system.dec_deg is None else system.dec_deg
    stellar_rv = sim_rv if system.stellar_rv_kms is None else system.stellar_rv_kms
    location = EarthLocation.from_geodetic(
        system.observatory_lon_deg * u.deg,
        system.observatory_lat_deg * u.deg,
        system.observatory_height_m * u.m,
    )
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    utc = Time(np.asarray(time_jd_utc), format="jd", scale="utc", location=location)
    light_time = utc.light_travel_time(target, kind="barycentric")
    bjd = np.asarray((utc.tdb + light_time).jd, dtype=float)
    berv = target.radial_velocity_correction("barycentric", obstime=utc).to_value(u.km / u.s)
    phase = ((bjd - system.transit_midpoint_bjd_tdb + 0.5 * system.period_days)
             % system.period_days) / system.period_days - 0.5
    half = 0.5 * system.transit_duration_hours / 24.0 / system.period_days
    in_transit = np.abs(phase) <= half
    # Box transit by default; retained as an explicit array for future limb/ingress weights.
    weight = in_transit.astype(float)
    return Orbit(name, float(ra), float(dec), float(stellar_rv), bjd, phase,
                 in_transit, weight, np.asarray(berv, dtype=float))
