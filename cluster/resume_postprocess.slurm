#!/bin/bash
#SBATCH --job-name=sharp-resume
#SBATCH --output=sharp-resume-%j.out
#SBATCH --error=sharp-resume-%j.err
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${SHARP_RESUME_SCRATCH:?Set SHARP_RESUME_SCRATCH to the retained job directory}"

IMAGE="${SHARP_IMAGE:-$HOME/wifi-doppler-preprocess.sif}"
RAW_ROOT="${SHARP_RAW_ROOT:-$SHARP_RESUME_SCRATCH/raw/CSI-80Mhz}"
OUTPUT_ROOT="${SHARP_OUTPUT_ROOT:-$HOME/doppler_traces_recomputed}"
PHASE_ROOT="$SHARP_RESUME_SCRATCH/phase"
PROCESSED_ROOT="$SHARP_RESUME_SCRATCH/processed"
EXCLUDED_ROOT="$SHARP_RESUME_SCRATCH/excluded_raw"

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
else
    CONTAINER_RUNTIME=singularity
fi

[[ -f "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
[[ -d "$RAW_ROOT/AR-1a" ]] || { echo "Missing copied raw data: $RAW_ROOT" >&2; exit 1; }

# PC2a_n00 is an empty placeholder (csi_buff shape [1, 256]), not a recording.
bad_raw="$RAW_ROOT/PC-2a/PC2a_n00.mat"
if [[ -f "$bad_raw" ]]; then
    mkdir -p "$EXCLUDED_ROOT/PC-2a"
    mv "$bad_raw" "$EXCLUDED_ROOT/PC-2a/"
fi
expected_recordings="$(
    find "$RAW_ROOT"/AR-* "$RAW_ROOT"/PC-* -maxdepth 1 -type f -name '*.mat' | wc -l
)"
[[ "$expected_recordings" -eq 203 ]] ||
    { echo "Expected 203 usable AR/PC recordings, found $expected_recordings" >&2; exit 1; }

[[ "$(find "$PHASE_ROOT" -name 'signal_*' | wc -l)" -eq 204 ]] ||
    { echo "Expected 204 signal files" >&2; exit 1; }
[[ "$(find "$PHASE_ROOT" -name 'Tr_vector_*' | wc -l)" -eq 816 ]] ||
    { echo "Expected 816 Tr_vector files" >&2; exit 1; }
[[ "$(find "$PROCESSED_ROOT" -name '*.mat' | wc -l)" -eq 816 ]] ||
    { echo "Expected 816 processed MAT files" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"

# A quota failure can leave one directly-written pickle truncated. Remove only
# unreadable or structurally invalid targets; valid partial output is retained.
"$CONTAINER_RUNTIME" exec --cleanenv --bind "$OUTPUT_ROOT:$OUTPUT_ROOT" "$IMAGE" \
    python -c '
import pickle, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
removed = []
for path in root.rglob("*.txt"):
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 100:
            raise ValueError("invalid shape %r" % (getattr(value, "shape", None),))
    except Exception as error:
        path.unlink()
        removed.append((path, str(error)))
print("removed corrupt targets:", len(removed))
for path, error in removed:
    print(path, error)
' "$OUTPUT_ROOT"

echo "resuming scratch=$SHARP_RESUME_SCRATCH"
echo "raw_root=$RAW_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "usable_recordings=$expected_recordings"

srun "$CONTAINER_RUNTIME" exec \
    --cleanenv \
    --bind "$RAW_ROOT:$RAW_ROOT:ro" \
    --bind "$SHARP_RESUME_SCRATCH:$SHARP_RESUME_SCRATCH" \
    --bind "$OUTPUT_ROOT:$OUTPUT_ROOT" \
    --pwd /opt/wifi-doppler-har \
    "$IMAGE" \
    python /opt/wifi-doppler-har/src/preprocessing/preprocess_sharp.py \
    --input-root "$RAW_ROOT" \
    --phase-root "$PHASE_ROOT" \
    --processed-root "$PROCESSED_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --subsets AR,PC \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --checkpoint-every 0

manifest="$OUTPUT_ROOT/manifest.json"
[[ -s "$manifest" ]] || { echo "Missing output manifest: $manifest" >&2; exit 1; }
"$CONTAINER_RUNTIME" exec --cleanenv --bind "$OUTPUT_ROOT:$OUTPUT_ROOT:ro" "$IMAGE" \
    python -c \
    'import json,sys; m=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); assert len(m["recordings"]) == expected; print("validated", expected, "recordings")' \
    "$manifest" "$expected_recordings"

echo "Recovery completed. Scratch retained at $SHARP_RESUME_SCRATCH"
