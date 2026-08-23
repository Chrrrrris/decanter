# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

Fast, pure-Python reduction of WINERED near-infrared echelle spectra. Currently
a WARP ([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) near-clone
validated across all three modes (HIRES-Y, HIRES-J, and WIDE).

## Install

```bash
pip install -e .
# For physical telluric/OH wavelength calibration:
pip install -e '.[wavecal]'
```

## Use

For a complete time series, the repository's `reduce.py` is the normal entry
point. It performs the original WARP-compatible cross-frame alignment and then
physical wavecal by default. HITRAN tables are downloaded by ExoJAX into a
per-user cache on first use; the raw-frame and calibration directories need no
line lists or cache files.

```bash
python reduce.py \
  --frames TOI2109/ \
  --listfile TOI2109.txt \
  --calib /data/TOI2109_calib \
  --out out/TOI2109 \
  --jobs 8 \
  --diagnostic-pdf
```

Without `--diagnostic-pdf`, wavecal still runs; only the report is skipped.
Select a concrete method with `--wavecal hybrid_refit`, `--wavecal
hybrid_static`, `--wavecal oh_refit`, or `--wavecal oh_static`. The default is
`--wavecal auto`. Use `--no-wavecal` for an intentionally WARP-only reduction.

```python
import decanter

calib = decanter.Calibration.from_dir("path/to/calibration_set")
r = decanter.reduce("obj.fits", calib, sky="sky.fits")   # nod-subtracted (A−B)

spec = r.obj[(1.30, 163)]        # order 163 at FSR cut 1.30
spec.wavelength, spec.flux       # vacuum-Å grid, flux
```

Omit `sky=` to reduce a single nod position on its own: `decanter.reduce("obj.fits", calib)`.
Without a sky frame there is no nod subtraction, so the background emission (OH
airglow lines), dark current, bias, and stray light are **retained** in the spectrum.
Pass `subtract_background=True` to estimate and remove that background from the
slit during extraction (suppressing the OH lines).

For a transit, `reduce_many` first applies the original WARP-compatible
cross-frame wavelength shift. Supplying `wavecal_config` layers the physical
telluric/OH hybrid correction on top; omitting it preserves WARP-only behavior.

```python
series = decanter.reduce_many(pairs, calib)  # original WARP alignment

series = decanter.reduce_many(
    pairs,
    calib,
    # auto: HIRES-Y/J -> hybrid_refit; WIDE -> hybrid_static
    wavecal_config=decanter.WavecalConfig(),
    wavecal_diagnostic_pdf="output/wavecal_diagnostics.pdf",  # optional
    workdir="output/corrected",
)

# Both calibration layers remain available independently.
series.shifts                    # WARP relative shifts, Angstrom
series.wavecal_solution.velocity # residual physical shifts, km/s/order
```

All four concrete modes remain selectable from the integrated `reduce.py`
command with `--wavecal`: `oh_static`, `oh_refit`, `hybrid_static`, and
`hybrid_refit`. The default `--wavecal auto` selects `hybrid_refit` for
HIRES-Y/J and `hybrid_static` for WIDE.

## Downstream HRCCS

Install the optional atmosphere and catalog dependencies, then run the
configuration-driven pipeline on a wavelength-calibrated Decanter directory:

```bash
pip install -e '.[wavecal,hrccs]'
decanter-hrccs examples/hrccs/wasp69b.toml
```

The pipeline reads the calibrated spectra and `telluric_transmission.npz`
written by the upstream reduction. The transmission product is continuous;
`telluric_threshold` is applied only downstream, so the mask can be changed
without repeating the reduction. Older Decanter products without this file can
still run, but no telluric pixels are masked.

The production atmosphere backend is ExoJAX. It obtains molecular lines through
the ExoJAX HITRAN interface, atomic lines from Kurucz, and equilibrium abundances
from FastChem. The raw-data directory does not need any of these databases.
First use may download them into `atmosphere.cache_dir`. The optional
`atmosphere.hitran_dir`, `atmosphere.cia_dir`, and `atmosphere.kurucz_dir`
fields point to existing shared line-list and collision-induced-absorption
caches.

For each species, the pipeline constructs one wide atmospheric model spanning
all retained orders, convolves it with the mode-specific Gaussian instrument
profile, and samples it on a log-wavelength grid with five points per
resolution FWHM. It then interpolates that common model onto each calibrated
order grid for the CCF analysis. Unless `[atmosphere].resolving_power` is set
explicitly, the pipeline reads `INSTMODE` from the Decanter products and uses
the WINERED nominal resolution: `R=28,000` for WIDE and `R=68,000` for
HIRES-Y/HIRES-J. The resolved mode and resolution are recorded in every
template and in `summary.json`.
The wide template is internally evaluated in bounded-memory chunks and only
the stitched wide product is cached. The defaults are 6,000 target-grid points
per molecular chunk and 2,000 per atomic chunk; lower
`atmosphere.wide_model_chunk_points` or
`atmosphere.atomic_wide_model_chunk_points` if memory is constrained.

The searched SVD rank is selected by the largest map S/N inside the configured
local window around the expected planet location. The exact expected-cell value
and unrestricted global maximum are also recorded, but do not select the rank.
The injection recovery and every null realization then use that fixed
observed-data-selected rank; they do not repeat the component search.
Test configurations default to five null realizations. Increase or decrease
this with `injection.null_realizations` in the TOML file.
Progress bars are enabled by default for template construction, SVD-rank
search, injection recovery, and null realizations. Set the top-level TOML field
`show_progress = false` to disable them in batch logs.
ExoJAX's low-level setup messages are hidden by default so they do not overwrite
the progress display; set `atmosphere.model_verbose = true` when debugging a
model build.
On macOS, FastChem chemistry runs in a cached, single-threaded subprocess so
its bundled OpenMP runtime cannot conflict with JAX/ExoJAX. No global
`KMP_DUPLICATE_LIB_OK` setting is needed.
By default the Kp grid spans zero to 1.5 times the expected Kp and the Vsys grid
spans at least five times the absolute stellar systemic velocity in each
direction. The output includes model, SVD-sequence, template-sequence,
rank-selection, and four-panel observed/injected/null diagnostic figures.

Example configurations for all validation datasets are in `examples/hrccs/`.
Edit the input/output paths as needed:

```bash
decanter-hrccs examples/hrccs/toi2109b.toml
decanter-hrccs examples/hrccs/wasp69b.toml
decanter-hrccs examples/hrccs/toi3486b.toml
```

### Fresh reductions to HRCCS: TOI-2109b (two nights) and WASP-193b

Yes: begin from the raw frames for each observing sequence, let `reduce.py`
apply the original Decanter/WARP extraction and alignment followed by physical
wavecal, and then run `decanter-hrccs` on that sequence's output directory.
Treat separate nights as separate reductions and separate HRCCS analyses. Do
not concatenate the two TOI-2109b nights before wavelength calibration or SVD;
their instrumental drift, telluric spectrum, noise, and optimal SVD rank are
night-specific.

From the Decanter repository root, the current local data layout can be reduced
with:

```bash
# TOI-2109b, take 1 (HIRES-Y -> auto selects hybrid_refit)
python reduce.py \
  --frames ../TOI2109 \
  --listfile ../TOI2109/TOI2109.txt \
  --calib ../TOI2109/2025_08_06/calibration_LCO25b_setting2_HIRES-Y100 \
  --out ../outputs/decanter_hrccs_inputs/toi2109b_take1 \
  --jobs 8 \
  --wavecal auto \
  --diagnostic-pdf

# TOI-2109b, take 2 (HIRES-Y -> auto selects hybrid_refit)
python reduce.py \
  --frames ../TOI2109_take2 \
  --listfile ../TOI2109_take2/TOI2109_take2.txt \
  --calib ../TOI2109_take2/2025_08_10/calibration_LCO25b_setting4_HIRES-Y100 \
  --out ../outputs/decanter_hrccs_inputs/toi2109b_take2 \
  --jobs 8 \
  --wavecal auto \
  --diagnostic-pdf

# WASP-193b (HIRES-Y -> auto selects hybrid_refit)
python reduce.py \
  --frames ../WASP193 \
  --listfile ../WASP193/WASP193.txt \
  --calib ../WASP193/2025_02_13/calibration_LCO25a_setting4_HIRES-Y100 \
  --out ../outputs/decanter_hrccs_inputs/wasp193b \
  --jobs 8 \
  --wavecal auto \
  --diagnostic-pdf
```

Use a new, empty `--out` directory for each run. `--overwrite` permits reuse
of a non-empty directory, but a fresh directory is safer for a science run.
After each reduction, confirm that the output root contains all of:

```text
warp_alignment.npz
wavecal_solution.npz
telluric_transmission.npz
wavecal_diagnostics.pdf       # only when --diagnostic-pdf was requested
```

The telluric product is essential for the configured downstream mask. If it is
absent, HRCCS warns and continues without masking telluric pixels; do not use
such a run as the production result.

Three matching HRCCS configurations are provided. Run them independently:

```bash
decanter-hrccs examples/hrccs/toi2109b_take1.toml
decanter-hrccs examples/hrccs/toi2109b_take2.toml
decanter-hrccs examples/hrccs/wasp193b.toml
```

The TOI-2109b configurations share the same literature system parameters but
have different Decanter input and HRCCS output directories. The WASP-193b
configuration adopts the Yee et al. (2025) system solution used by the local
WASP-193b notebook. Before a definitive run, review the species list, cloud-top
pressure, ephemeris, stellar systemic velocity, searched grids, and number of
null realizations. Five nulls are suitable only for an end-to-end smoke test;
a false-alarm probability intended for scientific interpretation needs many
more realizations.

The downstream HRCCS pipeline now uses the minimal-processing notebook method
by default for every target: linear uncentered SVD of the flux cube, an exact
SVD refit of the template injected multiplicatively into the low-rank scaling
cube, fixed-interior Pearson CCFs, and equal-order summation. The
`absolute_depth` template includes wavelength-dependent continuum opacity; the
SVD removes its constant component.

The WASP-69b notebook-matching validation profile is also included:

```bash
decanter-hrccs examples/hrccs/wasp69b_matched_validation.toml
```

The earlier projected-log implementation remains available only as an explicit
legacy/experimental configuration (`analysis_mode = "projected_log"`,
`template_signal = "differential"`, and `order_combination = "information"`);
it is not used by the standard examples or by default API calls.

`decanter.combine(series)` detects the per-exposure physical WCS differences
and resamples onto the first reduction's corrected grid before stacking.
