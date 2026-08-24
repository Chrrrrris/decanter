"""Unit tests for the optional post-wavecal SERVAL diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from decanter.serval import (
    TransitEphemeris,
    _adapter_text,
    _match_serval_rows,
    _retain_usable_orders,
    _telluric_mask,
    _transit_selection,
    _two_bins,
    robust_scatter,
    run_serval_rv_stability,
)
from decanter.wavecal.series import Series


def _series() -> Series:
    wave = np.column_stack((np.linspace(10_000.0, 10_010.0, 12),
                            np.linspace(10_100.0, 10_110.0, 12)))
    return Series(
        frame_ids=("a", "b"), orders=(160, 161), wave=wave,
        dv_pix_kms=np.ones(2), obj=np.ones((2, 12, 2)), sky=None,
        noise_fraction=np.full((2, 2), 0.01), time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([1.0, 2.0]), airmass=np.ones(2),
        instmode="HIRES-Y", fsr_cut=1.3, meta=[{}, {}],
    )


def test_robust_scatter_is_mad_sigma() -> None:
    assert robust_scatter([0.0, 1.0, 2.0]) == pytest.approx(1.4826)


def test_two_chronological_bins_use_inverse_variance_weights() -> None:
    time = np.asarray([4.0, 1.0, 3.0, 2.0])
    value = np.asarray([40.0, 10.0, 30.0, 20.0])
    error = np.asarray([1.0, 1.0, 2.0, 1.0])
    bin_time, bin_value, bin_error, count = _two_bins(time, value, error)
    np.testing.assert_allclose(bin_time, [1.5, 3.5])
    # First raw bin is 15; second is inverse-variance mean 38. The two-bin
    # display is then centered on their mean.
    np.testing.assert_allclose(bin_value, [-11.5, 11.5])
    np.testing.assert_allclose(bin_error, [1 / np.sqrt(2), 1 / np.sqrt(1.25)])
    np.testing.assert_array_equal(count, [2, 2])


def test_oot_bins_are_split_across_excluded_transit() -> None:
    time = np.asarray([1.0, 1.1, 1.9, 2.0])
    value = np.asarray([10.0, 20.0, 30.0, 40.0])
    error = np.ones(4)
    bin_time, bin_value, _, count = _two_bins(
        time, value, error, excluded_window=np.asarray([1.2, 1.8])
    )
    np.testing.assert_allclose(bin_time, [1.05, 1.95])
    np.testing.assert_allclose(bin_value, [-10.0, 10.0])
    np.testing.assert_array_equal(count, [2, 2])


def test_complete_serval_table_matches_in_order_with_duplicate_bjd() -> None:
    bjd = np.asarray([2_460_000.1, 2_460_000.2, 2_460_000.2])
    table = np.zeros((3, 5))
    table[:, 0] = bjd
    exposure, rows = _match_serval_rows(table, bjd)
    np.testing.assert_array_equal(exposure, [0, 1, 2])
    np.testing.assert_array_equal(rows, [0, 1, 2])


def test_transit_selection_uses_periodic_bjd_ephemeris() -> None:
    ephemeris = TransitEphemeris(
        period_days=2.0,
        transit_midpoint_bjd_tdb=2_460_000.0,
        transit_duration_hours=2.0,
    )
    bjd = np.asarray([2_460_001.94, 2_460_001.97, 2_460_002.00,
                      2_460_002.03, 2_460_002.06])
    utc = bjd - 0.003
    in_transit, window = _transit_selection(bjd, utc, ephemeris)
    np.testing.assert_array_equal(in_transit, [False, True, True, True, False])
    np.testing.assert_allclose(
        window,
        [2_460_002.0 - 0.003 - 1 / 24, 2_460_002.0 - 0.003 + 1 / 24],
    )


def test_ephemeris_reads_hrccs_system_table(tmp_path) -> None:
    path = tmp_path / "target.toml"
    path.write_text(
        "[system]\nperiod_days=3.0\ntransit_midpoint_bjd_tdb=2459000.0\n"
        "transit_duration_hours=2.5\n"
    )
    ephemeris = TransitEphemeris.from_toml(path)
    assert ephemeris.period_days == 3.0
    assert ephemeris.transit_duration_hours == 2.5


def test_telluric_product_is_mapped_and_dilated(tmp_path) -> None:
    series = _series()
    transmission = np.ones((2, 2, 12))
    transmission[1, 0, 6] = 0.90
    wavelength = np.repeat(series.wave.T[None, :, :], 2, axis=0)
    product = tmp_path / "telluric_transmission.npz"
    np.savez(
        product, orders=np.asarray(series.orders), wavelength_angstrom=wavelength,
        transmission=transmission,
    )
    mask = _telluric_mask(product, series, 0.995)
    assert np.all(mask[0, 3:10])
    assert not np.any(mask[1])


def test_serval_adapter_uses_all_orders_and_nominal_resolution() -> None:
    text, order_set = _adapter_text(26, 2550, 68_000.0)
    assert order_set == "0:26"
    assert 'R = 68000.0' in text
    assert 'iomax = 26' in text
    assert 'pmin = 164' in text
    assert 'pmax = 2386' in text


def test_orders_without_enough_unmasked_interior_are_excluded() -> None:
    n_pixels = 600
    series = Series(
        frame_ids=("a", "b"), orders=(160, 161, 162),
        wave=np.column_stack(tuple(
            np.linspace(10_000.0 + 100 * index, 10_010.0 + 100 * index, n_pixels)
            for index in range(3)
        )),
        dv_pix_kms=np.ones(3), obj=np.ones((2, n_pixels, 3)), sky=None,
        noise_fraction=np.full((2, 3), 0.01), time_jd=np.asarray([1.0, 2.0]),
        sky_time_jd=np.asarray([1.0, 2.0]), airmass=np.ones(2),
        instmode="HIRES-Y", fsr_cut=1.3, meta=[{}, {}],
    )
    mask = np.zeros((3, n_pixels), dtype=bool)
    mask[0] = True
    retained, retained_mask, keep = _retain_usable_orders(series, mask)
    assert retained.orders == (161, 162)
    assert retained.obj.shape == (2, n_pixels, 2)
    assert retained_mask.shape == (2, n_pixels)
    np.testing.assert_array_equal(keep, [False, True, True])


def test_serval_step_requires_physical_wavecal(tmp_path) -> None:
    transit = SimpleNamespace(wavecal_solution=None)
    with pytest.raises(ValueError, match="physical wavelength calibration"):
        run_serval_rv_stability(
            transit,
            tmp_path,
            ephemeris=TransitEphemeris(1.0, 2_460_000.0, 2.0),
        )
