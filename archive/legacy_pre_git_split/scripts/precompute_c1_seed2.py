"""Compute seed2 in isolated output directories, publish complete runs only.

This changes scheduling/output location only. The original worker will reuse
the complete checkpoint. Never replace a run already written by that worker.
"""
import argparse,os,shutil
from pathlib import Path
import train_c1_four_objectives as experiment

parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=experiment.ARMS,required=True);args=parser.parse_args()
official=experiment.OUT;shadow=official/'parallel_seed2'/args.arm
shadow.mkdir(parents=True,exist_ok=True);(shadow/'runs').mkdir(exist_ok=True)
shutil.copyfile(official/'protocol.json',shadow/'protocol.json')
experiment.OUT=shadow
experiment.train(args.arm,2)
source=shadow/'runs'/f'{args.arm}_seed2';target=official/'runs'/source.name
try:
    os.rename(source,target)
    print('PUBLISHED',target,flush=True)
except OSError:
    if not target.exists():raise
    print('Official worker already owns this run; shadow result not used.',flush=True)
