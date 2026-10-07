#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1'
sys.path.insert(0,str(ROOT))
from diagnostics.orthoflow3_q_learnability_v2 import run_q_rollouts as runner
runner.HERE=HERE
runner.main()
