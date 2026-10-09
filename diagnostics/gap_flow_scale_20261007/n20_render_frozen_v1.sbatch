#!/bin/bash
#SBATCH --job-name=gap1_n20_videos
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_video_v1_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
root=diagnostics/gap_flow_scale_20261007
videos="$root/n20_frozen_v1_videos"
mkdir -p "$videos"
case "$SLURM_ARRAY_TASK_ID" in
  0)
    trace="$root/n20_accept_v10_s12000/one_way_0000/traces/fresh_control_0000.npz"
    output="$videos/n20_one_way_complete_success.mp4"
    label=one_way_complete_success
    ;;
  1)
    trace="$root/n20_opposing_frozen_v1/case_0000/traces/fresh_control_0000.npz"
    output="$videos/n20_clean_bounded_gridlock.mp4"
    label=clean_bounded_gridlock
    ;;
  2)
    trace="$root/n20_opposing_frozen_v1/case_0018/traces/fresh_control_0018.npz"
    output="$videos/n20_partial_throughput_timeout.mp4"
    label=partial_throughput_timeout
    ;;
  *)
    echo "Invalid video index: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.render_scale_video \
  --trace "$trace" --output "$output" --label "$label"
