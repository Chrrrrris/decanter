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

To calibrate an existing Decanter reduction directory without repeating the
extraction, use the command-line layer. The report flag is optional and may be
given with no filename, in which case it writes `wavecal_diagnostics.pdf`
inside the output directory.

```bash
python scripts/run_wavecal.py \
  /data/decanter_reductions/wasp69b \
  /data/decanter_wavecal/wasp69b \
  --diagnostic-pdf
```

All four concrete modes remain selectable with `--mode`: `oh_static`,
`oh_refit`, `hybrid_static`, and `hybrid_refit`. See
`examples/run_wavecal_three_datasets.sh` for a complete three-dataset loop.

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

The searched SVD rank is selected by the largest map S/N inside the configured
local window around the expected planet location. The exact expected-cell value
and unrestricted global maximum are also recorded, but do not select the rank.
Null realizations repeat that rank search when estimating the false-alarm
fraction, while the displayed null map uses the observed rank for an
apples-to-apples comparison.
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

`decanter.combine(series)` detects the per-exposure physical WCS differences
and resamples onto the first reduction's corrected grid before stacking.
