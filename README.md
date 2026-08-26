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

For a time series, `reduce_many` applies the WARP cross-frame wavelength shift.
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

To replace WARP alignment with the explicitly enabled atmospheric registration
followed by physical calibration, use:

```bash
python reduce.py \
  --frames TOI2109/ --listfile TOI2109.txt \
  --calib TOI2109/calibration_LCO25b_setting2_HIRES-Y100 \
  --out output/TOI2109_atmospheric \
  --alignment atmospheric --wavecal auto --diagnostic-pdf
```

The four wavecal choices
remain `oh_static`, `oh_refit`, `hybrid_static`, and `hybrid_refit`.
`--alignment none` is available as an unregistered control.

`python reduce.py --help` reduces a whole night from raw frames and calibration
directory in one command. `decanter-hrccs examples/hrccs/wasp69b.toml` runs the
downstream cross-correlation pipeline on a calibrated output directory.

## RV stability check

`--serval-rv` runs SERVAL on the calibrated spectra and writes an RV-stability
plot, so a night can be checked without a separate analysis step. It needs an
event ephemeris, because in-transit or in-eclipse exposures are excluded before
SERVAL builds its template:

```bash
python reduce.py ... --serval-rv --serval-ephemeris examples/hrccs/wasp69b.toml
```

SERVAL itself is a separate program ([mzechmeister/serval](https://github.com/mzechmeister/serval)).
Point Decanter at a checkout with `--serval-dir`, or `$SERVAL`; `~/mzechmeister/serval`
and `~/serval` are tried otherwise.

`--serval-telluric-threshold` (default 0.90) masks fitted transmission below
that value. The default is measured rather than assumed, and it matters: on a
water-rich HIRES-J night, masking at 0.995 flags most of the spectrum and drops
half the orders for lack of surviving pixels, while masking nothing lets
saturated line cores bias the RVs. Both cost a factor of three in RV scatter.

The same check runs on an already-written reduction, which is how to re-run it
at a different threshold without repeating the reduction:

```python
import decanter

result = decanter.run_serval_rv_stability_directory(
    "output/corrected", ephemeris="examples/hrccs/wasp69b.toml",
)
print(result.exposure_rms_mps, result.figure_pdf)
```

Products land in `<out>/serval_rv_stability/`: the figure as PDF and PNG,
per-exposure RVs as CSV and NPZ, a JSON summary, and SERVAL's own output and
log. The figure shows every out-of-event exposure, two coarse time bins, and
the scatter of the RVs binned by exposure count against the `N^-1/2` a white
residual would follow.

## Transit and eclipse HRCCS

Set `observation_type = "transit"` or `"eclipse"` in the TOML `[system]`
table. New configurations should use the generic `event_midpoint_bjd_tdb` and
`event_duration_hours` fields; `transit_*` names are also supported.

For transmission, the signal spectra are the in-transit exposures. For an
eclipse sequence, the planet is visible out of eclipse, so those exposures are
the signal set and the in-eclipse spectra are excluded from the Kp--Vsys sum.
The eclipse template is deliberately only a species-presence proxy: Decanter
builds the same isothermal equilibrium transmission calculation and reverses
its line contrast. This does not claim a physically accurate dayside P--T
profile.

`[injection].scale` multiplies only the synthetic planet used in the single
injection-recovery experiment. It does not change the observed-data CCF,
component selection, or null trials. This is useful for a massive, low-scale-
height object: `scale = 10.0` asks how recoverable a ten-times-stronger line
contrast would be while leaving the observed species test unchanged.

```bash
decanter-hrccs examples/hrccs/bd143065b.toml
```

