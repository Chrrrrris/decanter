"""Isothermal equilibrium ExoJAX transmission templates."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import urllib.request
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

C_KMS = 299_792.458
G_CGS = 6.67430e-8
M_JUP_G = 1.89813e30
R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10
K_B_CGS = 1.380649e-16


@dataclass(frozen=True)
class Template:
    species: str
    wavelength_um: np.ndarray
    transit_depth: np.ndarray
    contrast: np.ndarray
    metadata: dict


def _download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)  # noqa: S310 - fixed scientific archives
    temporary.replace(path)
    return path


def _fastchem_files(config) -> tuple[Path, Path]:
    root = Path(config.cache_dir).expanduser() / "fastchem" / "input"
    abundance = (Path(config.fastchem_abundance_file).expanduser()
                 if config.fastchem_abundance_file else root / "element_abundances/asplund_2021.dat")
    logk = (Path(config.fastchem_logk_file).expanduser()
            if config.fastchem_logk_file else root / "logK/logK.dat")
    base = "https://raw.githubusercontent.com/NewStrangeWorlds/FastChem/master/input"
    if not abundance.exists():
        _download(f"{base}/element_abundances/asplund_2021.dat", abundance)
    if not logk.exists():
        _download(f"{base}/logK/logK.dat", logk)
    return abundance, logk


def _equilibrium_profiles_local(config, metallicity_dex: float, pressure_bar: np.ndarray,
                                temperature_k: np.ndarray, species: tuple[str, ...]):
    """Run FastChem locally; called only inside the isolated worker process."""
    import pyfastchem

    abundance_path, logk_path = _fastchem_files(config)
    solver = pyfastchem.FastChem(str(abundance_path), str(logk_path), 1)
    try:
        solver.setVerboseLevel(0)
    except Exception:
        pass
    abundances = np.asarray(solver.getElementAbundances(), dtype=float)
    for index in range(solver.getElementNumber()):
        if solver.getElementSymbol(index) not in ("H", "He"):
            abundances[index] *= 10.0 ** metallicity_dex
    solver.setElementAbundances(abundances)
    indices = {}
    for name in species:
        candidates = [name, name.replace("+", "") + "1" + "+" * name.count("+")]
        for candidate in candidates:
            try:
                candidate = solver.convertToHillNotation(candidate)
            except Exception:
                pass
            idx = solver.getGasSpeciesIndex(candidate)
            if idx != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
                indices[name] = int(idx)
                break
        if name not in indices:
            raise ValueError(f"FastChem does not provide equilibrium abundance for {name!r}")
    number = {name: np.full(pressure_bar.size, np.nan) for name in species}
    mmw = np.full(pressure_bar.size, np.nan)
    for layer, (pressure, temperature) in enumerate(zip(pressure_bar, temperature_k)):
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = [float(temperature)]
        input_data.pressure = [float(pressure)]
        flag = solver.calcDensities(input_data, output_data)
        if int(np.max(np.asarray(flag))) != 0:
            raise RuntimeError(f"FastChem failed in layer {layer}: flag={flag}")
        densities = np.asarray(output_data.number_densities[0], dtype=float)
        for name, idx in indices.items():
            number[name][layer] = densities[idx]
        mmw[layer] = float(output_data.mean_molecular_weight[0])
    total = pressure_bar * 1.0e6 / (K_B_CGS * temperature_k)
    vmr = {name: np.clip(values / total, 0.0, None) for name, values in number.items()}
    return vmr, mmw, {
        "engine": "pyfastchem equilibrium",
        "metallicity_dex": metallicity_dex,
        "abundance_file": str(abundance_path),
        "logk_file": str(logk_path),
        "median_vmr": {name: float(np.nanmedian(values)) for name, values in vmr.items()},
    }


def equilibrium_profiles(config, metallicity_dex: float, pressure_bar: np.ndarray,
                         temperature_k: np.ndarray, species: tuple[str, ...]):
    """Run FastChem outside the ExoJAX process to isolate OpenMP runtimes."""
    payload = {
        "config": {
            "cache_dir": str(config.cache_dir),
            "fastchem_abundance_file": config.fastchem_abundance_file,
            "fastchem_logk_file": config.fastchem_logk_file,
        },
        "metallicity_dex": float(metallicity_dex),
        "species": list(species),
    }
    with tempfile.TemporaryDirectory(prefix="decanter-fastchem-") as temporary:
        root = Path(temporary)
        request = root / "request.npz"
        response = root / "response.npz"
        np.savez_compressed(
            request,
            pressure_bar=np.asarray(pressure_bar, dtype=float),
            temperature_k=np.asarray(temperature_k, dtype=float),
            payload_json=np.asarray(json.dumps(payload, sort_keys=True)),
        )
        command = [sys.executable, "-m", "decanter.hrccs.fastchem_worker",
                   str(request), str(response)]
        worker_environment = os.environ.copy()
        # The macOS pyFastChem wheel bundles libomp in addition to NumPy's
        # runtime. Keep its compatibility escape hatch confined to this
        # single-threaded subprocess; never expose it to JAX/ExoJAX.
        worker_environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        worker_environment["OMP_NUM_THREADS"] = "1"
        worker_environment["OPENBLAS_NUM_THREADS"] = "1"
        worker_environment["MKL_NUM_THREADS"] = "1"
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
            env=worker_environment,
        )
        if completed.returncode != 0 or not response.exists():
            details = "\n".join(
                value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
            )
            raise RuntimeError(
                f"isolated FastChem worker failed with exit code {completed.returncode}:\n{details}"
            )
        with np.load(response, allow_pickle=False) as result:
            names = tuple(str(value) for value in result["species"])
            matrix = np.asarray(result["vmr"], dtype=float)
            vmr = {name: matrix[index] for index, name in enumerate(names)}
            mmw = np.asarray(result["mean_molecular_weight"], dtype=float)
            metadata = json.loads(str(result["metadata_json"]))
    if (not np.all(np.isfinite(mmw))) or np.any(mmw <= 0):
        raise RuntimeError("isolated FastChem worker returned an invalid mean molecular weight")
    for name, profile in vmr.items():
        if not np.all(np.isfinite(profile)) or np.any(profile < 0):
            raise RuntimeError(f"isolated FastChem worker returned an invalid {name} profile")
    metadata["execution"] = "isolated subprocess (OpenMP runtime separation)"
    return vmr, mmw, metadata


def _is_atomic(species: str) -> bool:
    bare = species.replace("+", "")
    return bare.isalpha() and len(bare) <= 2 and bare[0].isupper()


def _atomic_parts(species: str) -> tuple[str, int]:
    return species.replace("+", ""), 1 + species.count("+")


def _patch_atomic_import() -> None:
    name = "exojax.database.core_atom.line_strength"
    if name in sys.modules:
        return
    from exojax.database.core_atom import io as core_atom_io
    from exojax.utils.constants import Tref_original, ccgs, hcperk

    module = types.ModuleType(name)
    def line_strength_atom(A, gupper, nu_lines, elower, QTref_284, QTmask, Irwin=False):
        qref = np.asarray([QTref_284[mask] for mask in QTmask], dtype=float)
        if Irwin:
            qref[np.where(QTmask == 76)[0]] = core_atom_io.partfn_Fe(Tref_original)
        return (-A * gupper * np.exp(-hcperk * elower / Tref_original)
                * np.expm1(-hcperk * nu_lines / Tref_original)
                / (8.0 * np.pi * ccgs * nu_lines**2 * qref))
    module.line_strength_atom = line_strength_atom
    sys.modules[name] = module


def _kurucz_xsmatrix(opa, temperature, pressure):
    """ExoJAX-version-compatible atomic LPF cross-section matrix."""
    import jax.numpy as jnp
    from jax import jit, vmap
    from exojax.database.core.broadening import doppler_sigma
    from exojax.database.core.line_strength import line_strength
    from exojax.database.core_atom.broadening import gamma_vald3
    from exojax.database.core_atom.pf import interp_QT_284
    from exojax.opacity.lpf.lpf import xsmatrix as xsmatrix_lpf
    from exojax.utils.constants import Tref_original

    adb = opa.mdb
    temperature = jnp.asarray(temperature)
    pressure = jnp.asarray(pressure)
    qt = vmap(interp_QT_284, (0, None, None))(
        temperature, adb.T_gQT, adb.gQT_284species
    )
    qr = qt[:, adb.QTmask] / adb.QTref_284[adb.QTmask]
    strengths = jit(vmap(line_strength, (0, None, None, None, 0, None)))(
        temperature, adb.logsij0, adb.nu_lines, adb.elower, qr, Tref_original
    )
    strengths = jnp.nan_to_num(strengths, nan=0.0, posinf=0.0, neginf=0.0)
    broadening = jit(vmap(gamma_vald3, (
        0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None,
    )))(
        temperature, pressure * adb.vmrH, pressure * adb.vmrHH, pressure * adb.vmrHe,
        adb.ielem, adb.iion, adb.dev_nu_lines, adb.elower, adb.eupper,
        adb.atomicmass, adb.ionE, adb.gamRad, adb.gamSta, adb.vdWdamp, 1.0,
    )
    sigma = jit(vmap(doppler_sigma, (None, 0, None)))(
        adb.nu_lines, temperature, adb.atomicmass
    )
    return xsmatrix_lpf(opa.opainfo, sigma, broadening, strengths)


def _kurucz_path(species: str, root: Path) -> Path:
    from exojax.database.core_atom.io import PeriodicTable

    symbol, ion = _atomic_parts(species)
    matches = np.where(PeriodicTable == symbol)[0]
    if matches.size != 1:
        raise ValueError(f"unknown atomic symbol {symbol!r}")
    filename = f"gf{int(matches[0]):02d}{ion - 1:02d}.all"
    return _download(
        f"https://kurucz.harvard.edu/linelists/gfall/{filename}", root / filename
    )


class TemplateFactory:
    def __init__(self, system, atmosphere):
        self.system = system
        self.config = atmosphere
        self.cache = Path(atmosphere.cache_dir).expanduser()
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.cache = Path(tempfile.gettempdir()) / "decanter-hrccs-cache"
            self.cache.mkdir(parents=True, exist_ok=True)
            self.config = replace(atmosphere, cache_dir=str(self.cache))
        self._chemistry = {}

    def _key(self, species, wave):
        payload = {
            "schema": 1, "species": species,
            "wave": [round(float(wave[0]), 8), round(float(wave[-1]), 8), len(wave)],
            "system": vars(self.system), "atmosphere": vars(self.config),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def build(self, species: str, wavelength_um: np.ndarray) -> Template:
        wave = np.asarray(wavelength_um, dtype=float)
        path = self.cache / "templates" / f"{species.replace('+', 'p')}_{self._key(species, wave)}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return Template(species, data["wavelength_um"], data["transit_depth"],
                                data["contrast"], json.loads(str(data["metadata_json"])))
        if self.config.backend == "analytic":
            template = self._analytic(species, wave)
        elif self.config.backend == "exojax":
            if self.config.model_verbose:
                template = self._exojax(species, wave)
            else:
                captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
                try:
                    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr), \
                            warnings.catch_warnings():
                        warnings.filterwarnings("ignore", module=r"exojax\..*")
                        template = self._exojax(species, wave)
                except Exception as exc:
                    details = "\n".join(
                        value.strip() for value in (
                            captured_stdout.getvalue(), captured_stderr.getvalue()
                        ) if value.strip()
                    )
                    suffix = f"\nCaptured model output:\n{details}" if details else ""
                    raise RuntimeError(f"ExoJAX template failed for {species}: {exc}{suffix}") from exc
        else:
            raise ValueError(f"unknown atmosphere backend {self.config.backend!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, wavelength_um=template.wavelength_um,
                            transit_depth=template.transit_depth, contrast=template.contrast,
                            metadata_json=np.asarray(json.dumps(template.metadata, sort_keys=True)))
        return template

    def _analytic(self, species, wave):
        # Deterministic smoke-test backend; production defaults to ExoJAX.
        seed = int.from_bytes(hashlib.sha256(species.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        centers = rng.uniform(wave[0], wave[-1], 24)
        depth = np.full(wave.size, (self.system.planet_radius_rjup * R_JUP_CM /
                                    (self.system.stellar_radius_rsun * R_SUN_CM)) ** 2)
        for center in centers:
            width = center * 2.0 / C_KMS
            depth += 4.0e-4 * np.exp(-0.5 * ((wave - center) / width) ** 2)
        return Template(species, wave, depth, -(depth - np.nanpercentile(depth, 10)),
                        {"backend": "analytic_test_only"})

    def _exojax(self, species, wave):
        import jax.numpy as jnp
        from exojax.opacity import OpaCIA, OpaDirect, OpaPremodit, OpaRayleigh
        from exojax.postproc.specop import SopInstProfile
        from exojax.rt import ArtTransPure
        from exojax.utils.grids import wavenumber_grid
        from exojax.utils.instfunc import resolution_to_gaussian_std

        pad = 350.0 / C_KMS
        nu_min = 1.0e4 / (wave[-1] * (1.0 + pad))
        nu_max = 1.0e4 / (wave[0] * (1.0 - pad))
        n_grid = max(512, int(np.ceil(3.0 * 200_000.0 * np.log(nu_max / nu_min))))
        n_grid += n_grid % 2
        xsmode = "lpf" if _is_atomic(species) else "premodit"
        nu, _, opa_resolution = wavenumber_grid(
            nu_min, nu_max, n_grid, xsmode=xsmode, wavelength_order="ascending"
        )
        art = ArtTransPure(pressure_top=self.config.pressure_top_bar,
                           pressure_btm=self.config.pressure_bottom_bar,
                           nlayer=self.config.n_layers, integration="simpson")
        temperature_np = np.full(self.config.n_layers, self.system.equilibrium_temperature_k)
        temperature = jnp.asarray(temperature_np)
        all_species = tuple(dict.fromkeys((species, "H2", "He")))
        chemistry_key = (species, tuple(all_species))
        if chemistry_key not in self._chemistry:
            self._chemistry[chemistry_key] = equilibrium_profiles(
                self.config, self.system.metallicity_dex,
                np.asarray(art.pressure), temperature_np, all_species
            )
        vmr, mmw_np, chemistry = self._chemistry[chemistry_key]
        mmw = jnp.asarray(mmw_np)
        radius = self.system.planet_radius_rjup * R_JUP_CM
        stellar_radius = self.system.stellar_radius_rsun * R_SUN_CM
        gravity_btm = G_CGS * self.system.planet_mass_mjup * M_JUP_G / radius**2
        gravity = art.gravity_profile(temperature, mmw, radius, gravity_btm)

        if _is_atomic(species):
            _patch_atomic_import()
            from exojax.database.kurucz.api import AdbKurucz
            kurucz_root = (Path(self.config.kurucz_dir).expanduser()
                           if self.config.kurucz_dir else self.cache / "kurucz")
            path = _kurucz_path(species, kurucz_root)
            adb = AdbKurucz(path, nurange=[nu_min, nu_max], margin=0.0,
                            crit=self.config.kurucz_line_strength_crit, gpu_transfer=True,
                            vmr_fraction=[0.0, 0.16, 0.84])
            if np.asarray(getattr(adb, "nu_lines", [])).size == 0:
                raise ValueError(f"Kurucz contains no {species} lines in this order")
            opa = OpaDirect(adb, nu, wavelength_order="ascending")
            xs = _kurucz_xsmatrix(opa, temperature, art.pressure)
            molmass = float(np.nanmedian(np.asarray(adb.atomicmass)))
            source = f"Kurucz {path.name} via ExoJAX"
            line_count = int(np.asarray(adb.nu_lines).size)
        else:
            from exojax.database.hitran.api import MdbHitran
            from exojax.database.multimol import database_path_hitran12
            hitran_root = (Path(self.config.hitran_dir).expanduser()
                           if self.config.hitran_dir else self.cache / "hitran")
            path = hitran_root / database_path_hitran12(species)
            path.parent.mkdir(parents=True, exist_ok=True)
            mdb = MdbHitran(path, nurange=[nu_min, nu_max], isotope=self.config.hitran_isotope,
                            gpu_transfer=False, inherit_dataframe=False,
                            crit=self.config.line_strength_crit,
                            Ttyp=self.system.equilibrium_temperature_k, engine="vaex")
            opa = OpaPremodit(mdb, nu, diffmode=0,
                              broadening_resolution={"mode": "manual", "value": 0.2},
                              auto_trange=(max(100.0, 0.7 * self.system.equilibrium_temperature_k),
                                           1.3 * self.system.equilibrium_temperature_k),
                              allow_32bit=True, wavelength_order="ascending")
            xs = opa.xsmatrix(temperature, art.pressure)
            molmass = float(mdb.molmass)
            source = f"HITRAN {species} via ExoJAX"
            line_count = int(np.asarray(mdb.nu_lines).size)
        species_mmr = jnp.asarray(vmr[species] * molmass / mmw_np)
        molecular_dtau = art.opacity_profile_xs(xs, species_mmr, molmass, gravity)

        continuum = jnp.zeros_like(molecular_dtau)
        if self.config.include_rayleigh:
            for molecule, mass in (("H2", 2.01588), ("He", 4.002602)):
                rayleigh = OpaRayleigh(nu, molecule).xsvector()
                continuum += art.opacity_profile_xs(
                    rayleigh, jnp.asarray(vmr[molecule] * mass / mmw_np), mass, gravity
                )
        if self.config.include_cia:
            from exojax.database.cia.api import CdbCIA
            cia_root = (Path(self.config.cia_dir).expanduser()
                        if self.config.cia_dir else self.cache / "cia")
            for filename, first, second in (
                ("H2-H2_2011.cia", "H2", "H2"),
                ("H2-He_2011.cia", "H2", "He"),
            ):
                cia_path = _download(f"https://hitran.org/data/CIA/{filename}",
                                     cia_root / filename)
                opa_cia = OpaCIA(CdbCIA(str(cia_path), nurange=nu), nu_grid=nu)
                continuum += art.opacity_profile_cia(
                    opa_cia.logacia_matrix(temperature), temperature,
                    jnp.asarray(vmr[first]), jnp.asarray(vmr[second]),
                    mmw[:, None], gravity,
                )
        if self.config.cloud_top_pressure_bar is not None:
            cloudy = jnp.asarray(np.asarray(art.pressure) >= self.config.cloud_top_pressure_bar)
            continuum += jnp.where(cloudy[:, None], 1.0e6, 0.0)
        baseline = (radius / stellar_radius) ** 2
        def absolute(dtau):
            return np.asarray(art.run(dtau, temperature, mmw, radius, gravity_btm)) * baseline
        high_depth = absolute(molecular_dtau + continuum)
        high_continuum = absolute(continuum)
        sop = SopInstProfile(nu, vrmax=500.0)
        beta = resolution_to_gaussian_std(self.config.resolving_power)
        target_nu = 1.0e4 / wave[::-1]
        depth = np.asarray(sop.sampling(sop.ipgauss(high_depth, beta), 0.0, target_nu))[::-1]
        cont = np.asarray(sop.sampling(sop.ipgauss(high_continuum, beta), 0.0, target_nu))[::-1]
        contrast = -(depth - cont)
        meta = {"backend": "exojax", "source": source, "line_count": line_count,
                "opa_resolution": float(opa_resolution), "chemistry": chemistry,
                "continuum": {"rayleigh": self.config.include_rayleigh,
                              "cia": self.config.include_cia,
                              "cloud_top_pressure_bar": self.config.cloud_top_pressure_bar},
                "baseline_transit_depth": baseline}
        return Template(species, wave, depth, contrast, meta)
