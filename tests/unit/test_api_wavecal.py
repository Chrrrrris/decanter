"""Integration of physical wavecal as a post-WARP pipeline layer."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from astropy.io import fits

import decanter
import decanter.api as api
from decanter._reduction import OrderSpectrum, Reduction
from decanter.api import TransitSeries


def _reduction(frame_id: str, hour: int) -> Reduction:
    orders = (159, 160, 161)
    flux = np.linspace(100.0, 200.0, 500).astype(np.float32)
    obj = {
        (1.30, order): OrderSpectrum(
            order=order,
            fsr_cut=1.30,
            flux=flux,
            crval1=10000.0 + 50.0 * (order - orders[0]),
            cdelt1=0.03,
            crpix1=1.0,
        )
        for order in orders
    }
    return Reduction(
        obj_name="TARGET",
        obj_path=None,
        sky_path=None,
        obj=obj,
        sky=None,
        meta={
            "OBJFRAME": frame_id,
            "DATE-OBS": "2026-01-01",
            "UT-STR": f"{hour:02d}:00:00",
            "UT-END": f"{hour:02d}:00:00",
            "INSTMODE": "HIRES-Y",
            "WAVSHIFT": 0.125,
        },
    )


def test_physical_wavecal_is_applied_after_and_preserves_warp_shifts(monkeypatch) -> None:
    reductions = [_reduction("WINA00000001", 1), _reduction("WINA00000002", 2)]
    warp_shifts = np.array([0.0, 0.125])
    warp_series = TransitSeries(reductions=reductions, shifts=warp_shifts, refid=0)

    solve_module = importlib.import_module("decanter.wavecal.solve")

    def fake_solve(reference, config, *, verbose, diagnostic_pdf):
        shape = (reference.n_frames, reference.n_orders)
        return decanter.WavecalSolution(
            frame_ids=reference.frame_ids,
            orders=reference.orders,
            velocity=np.full(shape, 1.5),
            source=np.full(shape, "telluric", dtype="U16"),
            bracketed=np.ones(shape, dtype=bool),
            mode=config.mode,
        )

    monkeypatch.setattr(solve_module, "solve", fake_solve)
    corrected = decanter.calibrate_wavelengths(
        warp_series, decanter.WavecalConfig(), verbose=False
    )

    np.testing.assert_array_equal(corrected.shifts, warp_shifts)
    assert corrected.wavecal_solution is not None
    original = reductions[0].obj[(1.30, 159)]
    calibrated = corrected.reductions[0].obj[(1.30, 159)]
    assert calibrated.crval1 < original.crval1
    assert calibrated.flux is original.flux
    assert corrected.reductions[0].meta["WAVSHIFT"] == 0.125
    assert corrected.reductions[0].meta["WAVECAL"] == "hybrid_refit"


def test_warp_only_series_remains_the_default() -> None:
    series = TransitSeries(
        reductions=[_reduction("WINA00000001", 1)],
        shifts=np.zeros(1),
        refid=0,
    )
    assert series.wavecal_solution is None


def test_reduce_many_invokes_physical_layer_only_when_requested(monkeypatch) -> None:
    reduction = _reduction("WINA00000001", 1)
    calls = []

    monkeypatch.setattr(api, "reduce", lambda *args, **kwargs: reduction)

    def fake_calibrate(series, config, *, verbose, diagnostic_pdf):
        calls.append((config, verbose, diagnostic_pdf))
        return series

    monkeypatch.setattr(api, "calibrate_wavelengths", fake_calibrate)
    cfg = decanter.WavecalConfig()

    api.reduce_many([(object(), None)], object(), align=False)
    assert calls == []

    api.reduce_many(
        [(object(), None)],
        object(),
        align=False,
        wavecal_config=cfg,
        wavecal_verbose=False,
        wavecal_diagnostic_pdf="diagnostics.pdf",
    )
    assert calls == [(cfg, False, "diagnostics.pdf")]


def test_reduce_many_parallelizes_only_the_independent_extraction(monkeypatch) -> None:
    reduction = _reduction("WINA00000001", 1)
    submitted = []

    class ImmediateFuture:
        def result(self):
            return reduction

    class ImmediatePool:
        def __init__(self, max_workers):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args, **kwargs):
            submitted.append((function, args, kwargs))
            return ImmediateFuture()

    monkeypatch.setattr(api, "ProcessPoolExecutor", ImmediatePool)
    result = api.reduce_many(
        [(object(), None), (object(), None)], object(), align=False, jobs=3
    )

    assert len(submitted) == 2
    assert all(call[0] is api.reduce for call in submitted)
    assert len(result.reductions) == 2


def test_transit_series_write_to_persists_final_wcs_and_solutions(tmp_path) -> None:
    reduction = _reduction("WINA00000001", 1)
    solution = decanter.WavecalSolution(
        frame_ids=("WINA00000001",),
        orders=(159, 160, 161),
        velocity=np.full((1, 3), 2.0),
        source=np.full((1, 3), "telluric", dtype="U16"),
        bracketed=np.ones((1, 3), dtype=bool),
    )
    corrected = solution.apply(reduction)
    series = TransitSeries(
        reductions=[corrected],
        shifts=np.asarray([0.125]),
        refid=0,
        wavecal_solution=solution,
    )

    series.write_to(tmp_path)

    spectrum_path = next((tmp_path / "WINA00000001").glob("*_m159_*VAC.fits"))
    with fits.open(spectrum_path) as hdul:
        assert hdul[0].header["WAVECAL"] == "hybrid_refit"
        assert hdul[0].header["WAVSHIFT"] == 0.125
        assert hdul[0].header["CRVAL1"] < reduction.obj[(1.30, 159)].crval1
    assert (tmp_path / "warp_alignment.npz").exists()
    assert (tmp_path / "wavecal_solution.npz").exists()


def test_reduce_many_writes_only_after_physical_layer(monkeypatch, tmp_path) -> None:
    reduction = _reduction("WINA00000001", 1)
    monkeypatch.setattr(api, "reduce", lambda *args, **kwargs: reduction)

    def fake_calibrate(series, config, *, verbose, diagnostic_pdf):
        updated = _reduction("WINA00000001", 1)
        updated.meta["WAVECAL"] = "test-final-layer"
        return TransitSeries(
            reductions=[updated], shifts=series.shifts, refid=series.refid
        )

    monkeypatch.setattr(api, "calibrate_wavelengths", fake_calibrate)
    api.reduce_many(
        [(object(), None)], object(), align=False,
        wavecal_config=decanter.WavecalConfig(), workdir=tmp_path,
    )

    spectrum_path = next((tmp_path / "WINA00000001").glob("*_m159_*VAC.fits"))
    with fits.open(spectrum_path) as hdul:
        assert hdul[0].header["WAVECAL"] == "test-final-layer"


def test_combine_regrids_different_physical_wavecal_wcs() -> None:
    """A corrected series must not average mismatched wavelength pixels."""
    first = _reduction("WINA00000001", 1)
    second = _reduction("WINA00000002", 2)
    key = (1.30, 159)
    reference = first.obj[key]
    shifted = OrderSpectrum(
        order=reference.order,
        fsr_cut=reference.fsr_cut,
        flux=(reference.wavelength + 0.03).astype(np.float32),
        crval1=reference.crval1 + 0.03,
        cdelt1=reference.cdelt1,
        crpix1=reference.crpix1,
    )
    first.obj[key].flux[:] = first.obj[key].wavelength.astype(np.float32)
    second.obj[key] = shifted

    combined = decanter.combine([first, second], cut=1.30, weight="uniform")

    expected = combined.obj[key].wavelength
    np.testing.assert_allclose(combined.obj[key].flux[1:], expected[1:], atol=2e-3)
