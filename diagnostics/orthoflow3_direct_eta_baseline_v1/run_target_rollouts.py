#!/usr/bin/env python3
"""Use the already-audited Q-v2 fixed-current-Flow runner in this audit namespace."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/zhihan/research/Basin_C1')
from diagnostics.orthoflow3_q_learnability_v2 import run_q_rollouts as runner
runner.HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_direct_eta_baseline_v1')
runner.main()
