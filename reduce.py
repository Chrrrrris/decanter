#!/usr/bin/env python3
"""Reduce a WINERED time series through WARP alignment and physical wavecal.

The input area needs only raw WINA FITS files, an object/sky listfile, and the
night's calibration directory. HITRAN tables are downloaded by ExoJAX on first
use into Decanter's per-user cache; no line-list or cache paths are required.

Example:
    python reduce.py --frames TOI2109/ --listfile TOI2109.txt \
        --calib <TOI2109-calib-dir> --out out/TOI2109 --jobs 8 \
        --diagnostic-pdf

Physical wavecal is enabled by default. Automatic mode selection uses
hybrid_refit for HIRES-Y/J and hybrid_static for WIDE. Pass --no-wavecal only
when an intentionally WARP-only product is wanted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import decanter
from decanter.wavecal.config import AUTO_MODE, MODES


def _pairs(frames: Path, listfile: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for number, line in enumerate(listfile.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(f"{listfile}:{number}: expected 'OBJECT SKY'")
        paths = []
        for name in fields[:2]:
            path = frames / name
            if path.suffix.lower() != ".fits":
                path = path.with_suffix(".fits")
            if not path.exists():
                raise FileNotFoundError(f"missing frame listed at line {number}: {path}")
            paths.append(path)
        pairs.append((paths[0], paths[1]))
    if not pairs:
        raise ValueError(f"no object/sky pairs found in {listfile}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames", type=Path, required=True,
                        help="directory containing raw WINA FITS frames")
    parser.add_argument("--listfile", type=Path, required=True,
                        help="OBJECT SKY pairs, one pair per line")
    parser.add_argument("--calib", type=Path, required=True,
                        help="calibration-set directory or WARP reduction root")
    parser.add_argument("--out", type=Path, required=True,
                        help="new output root for the fully calibrated series")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel extraction workers (default: 1)")
    parser.add_argument(
        "--wavecal", "--wavecal-mode", dest="wavecal_mode",
        choices=(AUTO_MODE, *MODES), default=AUTO_MODE,
        help="default: auto (HIRES-Y/J=hybrid_refit, WIDE=hybrid_static)",
    )
    parser.add_argument(
        "--diagnostic-pdf", type=Path, nargs="?", const=Path("__AUTO__"),
        help="write the optional wavecal report; default path is OUT/wavecal_diagnostics.pdf",
    )
    parser.add_argument("--no-wavecal", action="store_true",
                        help="write an intentionally WARP-only reduction")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into an existing non-empty output directory")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.no_wavecal and args.diagnostic_pdf is not None:
        parser.error("--diagnostic-pdf requires physical wavecal")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        parser.error(f"--out is not empty: {args.out}; pass --overwrite to reuse it")

    pairs = _pairs(args.frames, args.listfile)
    calibration = decanter.Calibration.from_dir(args.calib)
    wavecal = None if args.no_wavecal else decanter.WavecalConfig(mode=args.wavecal_mode)
    diagnostic = args.diagnostic_pdf
    if diagnostic == Path("__AUTO__"):
        diagnostic = args.out / "wavecal_diagnostics.pdf"

    state = "disabled (WARP-only)" if wavecal is None else f"enabled ({args.wavecal_mode})"
    print(f"{len(pairs)} pairs; extraction jobs={args.jobs}; physical wavecal {state}")
    series = decanter.reduce_many(
        pairs,
        calibration,
        jobs=args.jobs,
        workdir=args.out,
        wavecal_config=wavecal,
        wavecal_diagnostic_pdf=diagnostic,
    )

    print(f"Done -> {args.out}")
    print(f"WARP alignment -> {args.out / 'warp_alignment.npz'}")
    if series.wavecal_solution is not None:
        print(
            f"Physical wavecal -> {args.out / 'wavecal_solution.npz'} "
            f"({series.wavecal_solution.mode})"
        )
    if diagnostic is not None:
        print(f"Diagnostic report -> {diagnostic}")


if __name__ == "__main__":
    main()
