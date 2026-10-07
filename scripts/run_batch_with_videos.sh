#!/usr/bin/env bash
# Run an experiment and fail delivery unless all four scenes have full videos.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 4 || "$3" != '--' ]]; then
  echo "Usage: $0 VIDEO_PLAN.json VIDEO_OUTPUT_DIR -- COMMAND [ARGS...]" >&2
  exit 2
fi
video_plan="$1"
video_output="$2"
shift 3
video_python="${C1_VIDEO_PYTHON:-$PWD/.venv-video/bin/python}"
# A failed experiment must never inherit a previous successful delivery marker.
rm -f -- "$video_output/manifest.json"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
"$video_python" -c 'import numpy; from PIL import Image'
"$@"
"$video_python" -m new_benchmark_common.batch_videos \
  --plan "$video_plan" --output "$video_output"
