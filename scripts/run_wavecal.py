#!/usr/bin/env python
"""Run Decanter's physical wavelength-calibration layer on reduced spectra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the adjacent checkout win over any older editable Decanter install when
# this file is invoked directly as ``python scripts/run_wavecal.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decanter.wavecal import (
    WavecalConfig,
    apply_solution_to_directory,
    load_series,
    solve,
    wavecal_report_pdf,
)
from decanter.wavecal.config import AUTO_MODE, MODES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer telluric/OH wavecal on existing WARP-aligned Decanter products."
    )
    parser.add_argument("input_dir", type=Path, help="per-frame Decanter reduction directory")
    parser.add_argument("output_dir", type=Path, help="corrected native-grid FITS directory")
    parser.add_argument(
        "--mode", choices=(AUTO_MODE, *MODES), default=AUTO_MODE,
        help="default: auto (Y/J=hybrid_refit, WIDE=hybrid_static)",
    )
    parser.add_argument("--fsr-cut", type=float, default=None)
    parser.add_argument("--linelist-dir", type=Path, default=None,
                        help="optional HITRAN cache override; ExoJAX downloads by default")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="optional opacity-cache override")
    parser.add_argument("--label", default=None, help="dataset label used in the PDF")
    parser.add_argument(
        "--diagnostic-pdf", type=Path, nargs="?", const=Path("__AUTO__"),
        help="write the full optional report; omit the path to write it in output_dir",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    series = load_series(args.input_dir, fsr_cut=args.fsr_cut)
    config = WavecalConfig(
        mode=args.mode,
        fsr_cut=args.fsr_cut,
        linelist_dir=str(args.linelist_dir) if args.linelist_dir else "",
        cache_dir=str(args.cache_dir) if args.cache_dir else "",
    ).resolved_for(series.instmode)
    print(f"{series.summary()}", flush=True)
    print(f"wavecal mode: {config.mode} (requested {args.mode})", flush=True)

    wants_report = args.diagnostic_pdf is not None
    result = solve(
        series, config, verbose=not args.quiet, return_diagnostics=wants_report
    )
    solution = result.solution if wants_report else result
    count = apply_solution_to_directory(
        solution, args.input_dir, args.output_dir, overwrite=args.overwrite
    )

    report_path = None
    if wants_report:
        report_path = (
            args.output_dir / "wavecal_diagnostics.pdf"
            if args.diagnostic_pdf == Path("__AUTO__") else args.diagnostic_pdf
        )
        label = args.label or args.input_dir.name
        wavecal_report_pdf(result, report_path, dataset=label)

    coverage = solution.coverage()
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "instmode": series.instmode,
        "requested_mode": args.mode,
        "resolved_mode": solution.mode,
        "frames": solution.n_frames,
        "orders": solution.n_orders,
        "fits_written": count,
        "finite_fraction": float(np.isfinite(solution.velocity).mean()),
        "coverage": coverage,
        "diagnostic_pdf": str(report_path.resolve()) if report_path else None,
    }
    summary_path = args.output_dir / "wavecal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {count} corrected FITS products to {args.output_dir}", flush=True)
    if report_path:
        print(f"wrote diagnostic report to {report_path}", flush=True)
    print(f"wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
