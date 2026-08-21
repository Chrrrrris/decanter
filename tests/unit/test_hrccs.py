from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from decanter.hrccs.analysis import ComponentResult, _map_summary, _select_component
from decanter.hrccs.config import (
    HRCCSConfig,
    InjectionConfig,
    InputConfig,
    SearchConfig,
    SystemConfig,
)
from decanter.wavecal.products import telluric_product


def _component(count, local_snr):
    empty = np.empty((0,))
    return ComponentResult(
        count=count,
        residual_cube=empty,
        filtered_template_cube=empty,
        exposure_ccf=empty,
        raw_map=empty,
        snr_map=empty,
        expected_snr=np.nan,
        local_peak_snr=local_snr,
        local_peak_kp_kms=np.nan,
        local_peak_vsys_kms=np.nan,
        peak_snr=np.nan,
        peak_kp_kms=np.nan,
        peak_vsys_kms=np.nan,
    )


def test_component_selection_uses_local_peak_not_expected_or_global():
    components = (_component(1, 2.1), _component(4, 5.7), _component(8, 3.2))
    assert _select_component(components).count == 4


def test_component_selection_ignores_nonfinite_local_peaks():
    assert _select_component((_component(1, np.nan), _component(2, 1.0))).count == 2
    with pytest.raises(ValueError, match="all SVD-component"):
        _select_component((_component(1, np.nan),))


def test_map_summary_separates_expected_local_and_global_maxima():
    kp = np.arange(0.0, 201.0, 10.0)
    vsys = np.arange(-50.0, 51.0, 5.0)
    values = np.zeros((kp.size, vsys.size))
    values[np.where(kp == 100)[0][0], np.where(vsys == 0)[0][0]] = 2.0
    values[np.where(kp == 110)[0][0], np.where(vsys == 5)[0][0]] = 4.0
    values[np.where(kp == 190)[0][0], np.where(vsys == -45)[0][0]] = 9.0
    summary = _map_summary(values, kp, vsys, 100.0, 0.0, 20.0, 10.0)
    assert summary == (2.0, 4.0, 110.0, 5.0, 9.0, 190.0, -45.0)


def test_default_search_grids_and_local_window():
    config = HRCCSConfig(
        input=InputConfig("products"),
        system=SystemConfig(
            period_days=1.0,
            transit_midpoint_bjd_tdb=2_460_000.0,
            transit_duration_hours=2.0,
            expected_kp_kms=200.0,
        ),
    )
    _, kp, vsys = config.grids(-10.0)
    assert kp[0] == 0.0
    assert kp[-1] == 300.0
    assert vsys[0] == -50.0
    assert vsys[-1] == 50.0
    assert config.search.local_kp_half_width_kms == 30.0
    assert config.search.local_vsys_half_width_kms == 15.0
    assert InjectionConfig().null_realizations == 5
    assert config.show_progress is True


def test_search_grid_steps_must_be_positive():
    config = HRCCSConfig(
        input=InputConfig("products"),
        system=SystemConfig(
            period_days=1.0,
            transit_midpoint_bjd_tdb=2_460_000.0,
            transit_duration_hours=2.0,
            expected_kp_kms=200.0,
        ),
    )
    with pytest.raises(ValueError, match="kp_step_kms"):
        replace(config, search=SearchConfig(kp_step_kms=0.0)).validate()
    with pytest.raises(ValueError, match="null_realizations"):
        replace(config, injection=InjectionConfig(null_realizations=0)).validate()


def test_wavecal_telluric_product_is_continuous_and_unthresholded(tmp_path):
    wave = np.array([[10_000.0, 10_100.0], [10_001.0, 10_101.0], [10_002.0, 10_102.0]])
    native = np.array([[0.99, 0.75], [0.85, 0.60], [0.98, 0.80]])
    series = SimpleNamespace(
        n_frames=2, n_orders=2, n_pixels=3, wave=wave,
        frame_ids=("a", "b"), orders=(42, 43),
    )
    model = SimpleNamespace(
        species=("H2O",), native_template=native,
        parameters=np.zeros((2, 6)), family=np.array(["gaussian", "gaussian"]),
        template_rms=np.array([0.01, 0.02]), meta={},
    )
    solution = SimpleNamespace(velocity=np.zeros((2, 2)), mode="hybrid_static")
    run = SimpleNamespace(
        telluric_model=model, series=series, solution=solution,
        telluric_refit_parameters=None, telluric_accepted=np.ones((2, 2), bool),
        telluric_peak=np.ones((2, 2)), _telluric_tau=None,
    )
    output = telluric_product(run, tmp_path / "telluric_transmission.npz")
    with np.load(output, allow_pickle=False) as product:
        assert product["transmission"].shape == (2, 2, 3)
        np.testing.assert_allclose(product["transmission"][0], native.T)
        assert np.any((product["transmission"] > 0.60)
                      & (product["transmission"] < 0.99))
