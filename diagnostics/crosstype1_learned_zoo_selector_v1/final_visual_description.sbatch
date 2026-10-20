#!/bin/bash
#SBATCH --job-name=crosstype1_final_visual_description
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/final_visual_description_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/describe_locked_timeouts.py
XDG_CACHE_HOME=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/cache TMPDIR=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/tmp MOZ_CRASHREPORTER_DISABLE=1 firefox --headless --no-remote --profile /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/visual_preview_profile --screenshot /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/locked_nominal_preview.png --window-size 720,790 file:///home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/ring_ippo_nominal_only_trajectory.svg
XDG_CACHE_HOME=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/cache TMPDIR=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/tmp MOZ_CRASHREPORTER_DISABLE=1 firefox --headless --no-remote --profile /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/visual_preview_profile --screenshot /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/locked_safety_preview.png --window-size 720,790 file:///home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/ring_ippo_safety_baseline_trajectory.svg
XDG_CACHE_HOME=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/cache TMPDIR=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/tmp MOZ_CRASHREPORTER_DISABLE=1 firefox --headless --no-remote --profile /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/visual_preview_profile --screenshot /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/locked_pipeline_preview.png --window-size 720,790 file:///home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/ring_ippo_pipeline_trajectory.svg
XDG_CACHE_HOME=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/cache TMPDIR=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/tmp MOZ_CRASHREPORTER_DISABLE=1 firefox --headless --no-remote --profile /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/visual_preview_profile --screenshot /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/locked_features_preview.png --window-size 2000,1980 file:///home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/figures/locked_feature_distributions.svg
