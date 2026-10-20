#!/bin/bash
#SBATCH --job-name=crosstype1_scale_visual_review
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/scale_visual_review_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
XDG_CACHE_HOME=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/cache TMPDIR=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/tmp MOZ_CRASHREPORTER_DISABLE=1 firefox --headless --no-remote --profile /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/visual_preview_profile --screenshot /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/scale_preview.png --window-size 1200,530 file:///home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/scale/scale_curve.svg
