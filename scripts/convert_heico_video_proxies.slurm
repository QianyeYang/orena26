#!/bin/bash
# Convert the 30 browser-incompatible HeiCo AVI files to seekable 480p H.264 MP4
# proxies. LapChole is not included because its source files are already H.264 MP4.
#
# Submit from the repository root:
#   sbatch scripts/convert_heico_video_proxies.slurm
#
# The array cap and per-task allocation keep at most four two-core encoders active.
# Existing valid proxies are skipped, so re-submission safely resumes an incomplete run.
#SBATCH --job-name=heico-480p-proxy
#SBATCH --partition=allgpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-29%4
#SBATCH --chdir=/datasets/engs2732/orena
#SBATCH --output=/datasets/engs2732/orena/logs/heico-proxy-%A_%a.log
#SBATCH --error=/datasets/engs2732/orena/logs/heico-proxy-%A_%a.log

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/datasets/engs2732/orena}"
SOURCE_DIR="$REPO/data/focus/heico/videos"
OUTPUT_DIR="$REPO/data/focus/heico/video-proxies-480p"
TARGET_MIB="${TARGET_MIB:-285}"
HEIGHT="${HEIGHT:-480}"
PRESET="${PRESET:-veryfast}"
CORES="${SLURM_CPUS_PER_TASK:-2}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

source /users/engs2732/miniconda3/etc/profile.d/conda.sh
conda activate orena
cd "$REPO"

# Prevent libraries surrounding FFmpeg from creating unallocated helper threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

command -v ffmpeg >/dev/null || {
    echo "ffmpeg is unavailable in the orena environment" >&2
    exit 2
}
command -v ffprobe >/dev/null || {
    echo "ffprobe is unavailable in the orena environment" >&2
    exit 2
}
ffmpeg -hide_banner -encoders 2>/dev/null \
    | grep '[[:space:]]libx264[[:space:]]' >/dev/null || {
    echo "the available FFmpeg build does not provide libx264" >&2
    exit 2
}

mapfile -d '' -t VIDEOS < <(
    find "$SOURCE_DIR" -maxdepth 1 -type f -iname '*.avi' -print0 | sort -z
)
if [[ "${#VIDEOS[@]}" -ne 30 ]]; then
    echo "expected 30 HeiCo AVI files, found ${#VIDEOS[@]} in $SOURCE_DIR" >&2
    exit 2
fi
if (( TASK_ID < 0 || TASK_ID >= ${#VIDEOS[@]} )); then
    echo "array index $TASK_ID is outside 0..$((${#VIDEOS[@]} - 1))" >&2
    exit 2
fi

source_video="${VIDEOS[$TASK_ID]}"
stem="$(basename "${source_video%.*}")"
mkdir -p "$OUTPUT_DIR"
output_video="$OUTPUT_DIR/$stem.mp4"
temporary_video="$OUTPUT_DIR/.$stem.${SLURM_ARRAY_JOB_ID:-$$}_${TASK_ID}.tmp.mp4"

valid_proxy() {
    local candidate="$1"
    local properties
    [[ -s "$candidate" ]] || return 1
    properties="$(
        ffprobe -v error -select_streams v:0 \
            -show_entries stream=codec_name,height,pix_fmt \
            -of csv=p=0 "$candidate" 2>/dev/null
    )"
    [[ "$properties" == "h264,$HEIGHT,yuv420p" ]]
}

if valid_proxy "$output_video"; then
    echo "skip: valid proxy already exists: $output_video"
    exit 0
fi
if [[ -e "$output_video" ]]; then
    echo "refusing to replace an existing invalid proxy: $output_video" >&2
    exit 3
fi

cleanup() {
    [[ ! -e "$temporary_video" ]] || rm -f -- "$temporary_video"
}
trap cleanup EXIT INT TERM

duration="$(
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$source_video"
)"
bitrate_kbps="$(
    awk -v mib="$TARGET_MIB" -v seconds="$duration" 'BEGIN {
        if (seconds <= 0) exit 2
        rate = int(mib * 1024 * 1024 * 8 / seconds / 1000)
        print (rate < 100 ? 100 : rate)
    }'
)"

echo "source=$source_video"
echo "output=$output_video"
echo "duration_seconds=$duration target_mib=$TARGET_MIB video_kbps=$bitrate_kbps"
echo "height=$HEIGHT fps=source preset=$PRESET threads=$CORES host=$(hostname)"

# A five-second maximum GOP makes remote seeking responsive. No -r option is
# supplied, so the original 25 fps is preserved. HeiCo has no audio streams.
ffmpeg -hide_banner -nostdin -y \
    -i "$source_video" \
    -map 0:v:0 -an -sn -dn \
    -vf "scale=-2:${HEIGHT}:flags=lanczos,setsar=1" \
    -filter_threads "$CORES" \
    -c:v libx264 -preset "$PRESET" -profile:v main -level:v 3.1 \
    -pix_fmt yuv420p -tag:v avc1 \
    -b:v "${bitrate_kbps}k" \
    -g 125 -keyint_min 25 \
    -threads:v "$CORES" \
    -movflags +faststart \
    "$temporary_video"

valid_proxy "$temporary_video" || {
    echo "FFmpeg completed but proxy validation failed: $temporary_video" >&2
    exit 4
}

source_duration="$duration"
proxy_duration="$(
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$temporary_video"
)"
awk -v source="$source_duration" -v proxy="$proxy_duration" 'BEGIN {
    delta = source - proxy
    if (delta < 0) delta = -delta
    tolerance = source * 0.001
    if (tolerance < 2) tolerance = 2
    if (delta > tolerance) {
        printf "duration mismatch: source=%.3f proxy=%.3f\n", source, proxy > "/dev/stderr"
        exit 1
    }
}'

mv -- "$temporary_video" "$output_video"
trap - EXIT INT TERM
size_mib="$(du -m "$output_video" | awk '{print $1}')"
echo "done: $output_video (${size_mib} MiB)"
