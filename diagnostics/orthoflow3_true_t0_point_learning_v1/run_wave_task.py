#!/usr/bin/env python3
"""Dispatch one frozen wave item to the per-state search runner."""
import argparse, json, subprocess
from pathlib import Path

HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_true_t0_point_learning_v1')
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11'

ap=argparse.ArgumentParser(); ap.add_argument('--wave',required=True); ap.add_argument('--index',type=int,required=True)
a=ap.parse_args(); item=json.load(open(HERE/f'{a.wave}.json'))['items'][a.index]
subprocess.run([PY,str(HERE/'run_state_search.py'),'--split',item['split'],'--rank',str(item['rank'])],check=True)
