"""Run a frozen gap-audit arm plan with the authoritative migration runner."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/zhihan/research/Basin_C1')
from diagnostics.orthoflow3_representation_migration_v1 import run_subset_arms as runner
runner.HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_zero_active_gap_connectivity_v1')
runner.main()
