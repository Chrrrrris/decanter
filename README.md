# decanter

[![tests](https://github.com/astroshrey/decanter/actions/workflows/tests.yml/badge.svg)](https://github.com/astroshrey/decanter/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/decanter/badge/?version=latest)](https://decanter.readthedocs.io/en/latest/)

Fast, pure-Python reduction of WINERED near-infrared echelle spectra. A WARP
([Hamano et al. 2024](https://arxiv.org/abs/2401.04876)) near-clone validated
across all three modes (HIRES-Y, HIRES-J, and WIDE), plus an absolute
wavelength calibration that anchors each order to telluric absorption and OH
airglow.

## Install

```bash
pip install -e .
pip install -e '.[wavecal]'   # for the physical wavelength calibration
```

## Use

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

For a transit, `reduce_many` applies the WARP cross-frame wavelength shift.
Passing `wavecal_config` layers the telluric/OH calibration on top of it;
omitting it leaves the WARP-only behaviour unchanged.

```python
series = decanter.reduce_many(pairs, calib)                # WARP alignment only

series = decanter.reduce_many(
    pairs, calib,
    wavecal_config=decanter.WavecalConfig(),               # hybrid, mode="auto"
    workdir="output/corrected",
)

series.shifts                     # WARP relative shifts, Å
series.wavecal_solution.velocity  # physical shifts, km/s per (frame, order)
```

The calibration is applied by rescaling each order's `CRVAL1`/`CDELT1`, so the
written spectra are calibrated without the flux being resampled.

The command-line pipeline defaults to WARP cross-frame alignment followed by
physical wavecal. To replace WARP alignment with a two-pass atmospheric
alignment, use:

```bash
python reduce.py \
  --frames TOI2109/ --listfile TOI2109.txt \
  --calib TOI2109/calibration_LCO25b_setting2_HIRES-Y100 \
  --out output/TOI2109_atmospheric \
  --alignment atmospheric --wavecal auto --diagnostic-pdf
```

This first searches a broad common velocity grid and pools standardized CCFs
from telluric-rich object orders and OH-rich paired-sky orders. Tellurics take
priority in orders rich in both references. The resulting time-variable shift
registers all orders in wavelength before the atmospheric templates are
rebuilt and the selected fine wavecal mode is run. The four wavecal choices
remain `oh_static`, `oh_refit`, `hybrid_static`, and `hybrid_refit`.
`--alignment none` is available as an unregistered control.

`python reduce.py --help` reduces a whole night from raw frames and calibration
directory in one command. `decanter-hrccs examples/hrccs/wasp69b.toml` runs the
downstream cross-correlation pipeline on a calibrated output directory.
