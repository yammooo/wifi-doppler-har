#!/bin/bash
#SBATCH --job-name=sharp-ar-pc
#SBATCH --output=sharp-ar-pc-%j.out
#SBATCH --error=sharp-ar-pc-%j.err
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

IMAGE="${SHARP_IMAGE:-$HOME/wifi-doppler-preprocess.sif}"
RAW_ROOT="${SHARP_RAW_ROOT:-$HOME/CSI-80Mhz}"
OUTPUT_ROOT="${SHARP_OUTPUT_ROOT:-$HOME/doppler_traces_recomputed}"
SUBSETS="${SHARP_SUBSETS:-AR,PC}"
SCRATCH_BASE="${SHARP_SCRATCH_BASE:-/ext/$USER/wifi-doppler-har}"
SCRATCH="$SCRATCH_BASE/$SLURM_JOB_ID"
MIN_SCRATCH_GB="${SHARP_MIN_SCRATCH_GB:-300}"

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME=singularity
else
    echo "Neither apptainer nor singularity is available" >&2
    exit 1
fi

[[ -f "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
[[ -d "$RAW_ROOT/AR-1a" ]] || { echo "Missing AR data under $RAW_ROOT" >&2; exit 1; }
[[ -d "$RAW_ROOT/PC-1a" ]] || { echo "Missing PC data under $RAW_ROOT" >&2; exit 1; }

available_kb="$(df -Pk /ext | awk 'NR == 2 {print $4}')"
required_kb="$((MIN_SCRATCH_GB * 1024 * 1024))"
if (( available_kb < required_kb )); then
    echo "Insufficient /ext space: need ${MIN_SCRATCH_GB} GiB free" >&2
    df -h /ext >&2
    exit 1
fi

mkdir -p "$SCRATCH/phase" "$SCRATCH/processed" "$OUTPUT_ROOT"

echo "job=$SLURM_JOB_ID host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
echo "image=$IMAGE"
echo "raw_root=$RAW_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "scratch=$SCRATCH"
echo "subsets=$SUBSETS"
echo "raw_mat_files=$(find "$RAW_ROOT" -type f -name '*.mat' | wc -l)"
du -sh "$RAW_ROOT"
df -h /ext "$OUTPUT_ROOT"

command=(
    "$CONTAINER_RUNTIME" exec
    --cleanenv
    --bind "$RAW_ROOT:$RAW_ROOT:ro"
    --bind "$SCRATCH:$SCRATCH"
    --bind "$OUTPUT_ROOT:$OUTPUT_ROOT"
    --pwd /opt/wifi-doppler-har
    "$IMAGE"
    python /opt/wifi-doppler-har/src/preprocessing/preprocess_sharp.py
    --input-root "$RAW_ROOT"
    --phase-root "$SCRATCH/phase"
    --processed-root "$SCRATCH/processed"
    --output-root "$OUTPUT_ROOT"
    --subsets "$SUBSETS"
    --jobs "$SLURM_CPUS_PER_TASK"
    --checkpoint-every 0
)

[[ -n "${SHARP_LIMIT:-}" ]] && command+=(--limit "$SHARP_LIMIT")
[[ -n "${SHARP_H_END:-}" ]] && command+=(--h-end "$SHARP_H_END")
[[ -n "${SHARP_DOPPLER_START:-}" ]] && command+=(--doppler-start "$SHARP_DOPPLER_START")
[[ -n "${SHARP_DOPPLER_END:-}" ]] && command+=(--doppler-end "$SHARP_DOPPLER_END")

srun "${command[@]}"

manifest="$OUTPUT_ROOT/manifest.json"
[[ -s "$manifest" ]] || { echo "Missing output manifest: $manifest" >&2; exit 1; }
"$CONTAINER_RUNTIME" exec --cleanenv --bind "$OUTPUT_ROOT:$OUTPUT_ROOT:ro" "$IMAGE" \
    python -c \
    'import json,sys; p=sys.argv[1]; m=json.load(open(p)); assert m["recordings"]; print("validated", len(m["recordings"]), "recordings in", p)' \
    "$manifest"

if [[ "${SHARP_KEEP_SCRATCH:-0}" == "1" ]]; then
    echo "Keeping scratch at $SCRATCH"
else
    case "$SCRATCH" in
        "/ext/$USER/wifi-doppler-har/$SLURM_JOB_ID") rm -rf -- "$SCRATCH" ;;
        *) echo "Refusing to remove unexpected scratch path: $SCRATCH" >&2; exit 1 ;;
    esac
fi

echo "SHARP preprocessing completed successfully"
