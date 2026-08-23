"""The operational reduce.py command enables series-level wavecal by default."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import decanter
import pytest


def _load_cli():
    path = Path(__file__).resolve().parents[2] / "reduce.py"
    spec = importlib.util.spec_from_file_location("decanter_reduce_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_runs_wavecal_and_optional_report_by_default(monkeypatch, tmp_path) -> None:
    cli = _load_cli()
    frames = tmp_path / "raw"
    frames.mkdir()
    (frames / "OBJ.fits").touch()
    (frames / "SKY.fits").touch()
    listfile = tmp_path / "pairs.txt"
    listfile.write_text("OBJ SKY\n")
    output = tmp_path / "output"
    captured = {}

    monkeypatch.setattr(decanter.Calibration, "from_dir", lambda path: object())

    def fake_reduce_many(pairs, calibration, **kwargs):
        captured["pairs"] = pairs
        captured["calibration"] = calibration
        captured.update(kwargs)
        return SimpleNamespace(wavecal_solution=SimpleNamespace(mode="hybrid_refit"))

    monkeypatch.setattr(decanter, "reduce_many", fake_reduce_many)
    monkeypatch.setattr(
        "sys.argv",
        [
            "reduce.py", "--frames", str(frames), "--listfile", str(listfile),
            "--calib", str(tmp_path / "calib"), "--out", str(output),
            "--jobs", "8", "--wavecal", "hybrid_refit", "--diagnostic-pdf",
        ],
    )

    cli.main()

    assert captured["jobs"] == 8
    assert captured["workdir"] == output
    assert captured["wavecal_config"].mode == "hybrid_refit"
    assert captured["wavecal_diagnostic_pdf"] == output / "wavecal_diagnostics.pdf"


def test_pairs_rejects_duplicate_science_frames_before_reduction(tmp_path) -> None:
    cli = _load_cli()
    frames = tmp_path / "raw"
    frames.mkdir()
    for name in ("OBJ.fits", "SKY1.fits", "SKY2.fits"):
        (frames / name).touch()
    listfile = tmp_path / "pairs.txt"
    listfile.write_text("OBJ SKY1\nOBJ SKY2\n")

    with pytest.raises(ValueError, match=r"OBJ\.fits on lines 1, 2"):
        cli._pairs(frames, listfile)
