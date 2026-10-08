#!/bin/bash
#SBATCH --job-name=gap1_n20_mirror_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_mirror_audit_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.audit_n50_mirror_equivariance \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --checkpoint diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1/snapshot_0006000.pkl \
  --checkpoint diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1/snapshot_0020000.pkl \
  --output diagnostics/gap_flow_scale_20261007/n20_mirror_audit_v1.json
