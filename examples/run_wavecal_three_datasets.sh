#!/usr/bin/env bash
set -euo pipefail

# Run from the Decanter repository root. With --mode omitted, HIRES-Y/J use
# hybrid_refit and WIDE uses hybrid_static. The PDF is optional; remove
# --diagnostic-pdf if only corrected FITS products and the NPZ/JSON are wanted.
workspace="${DECANTER_VALIDATION_ROOT:-/path/to/workspace}"
for dataset in toi2109b wasp69b toi3486b; do
  python scripts/run_wavecal.py \
    "$workspace/outputs/decanter_reductions/$dataset" \
    "$workspace/outputs/decanter_wavecal/$dataset" \
    --diagnostic-pdf \
    --label "$dataset"
done
