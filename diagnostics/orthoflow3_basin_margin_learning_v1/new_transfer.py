#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/zhihan/research/Basin_C1')
HERE = ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1'
SOURCE = ROOT/'diagnostics/orthoflow3_large_margin_ball_transfer_v1/transfer_audit.py'
BALL = HERE/'eligible_new_anchor_source'
OUT = HERE/'new_anchor_transfers'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    if fields is None:
        fields = list(rows[0])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def aggregate_eligible() -> list[str]:
    BALL.mkdir(parents=True, exist_ok=True)
    states=[]; centers=[]; balls=[]; inside=[]; eligible=[]
    for directory in sorted((HERE/'anchor_builds').iterdir()):
        if not (directory/'conservative_ball_parameters.csv').exists():
            continue
        state_manifest=json.load(open(directory/'frozen_state_manifest.json'))['states'][0]
        ball=list(csv.DictReader(open(directory/'conservative_ball_parameters.csv')))[0]
        center=list(csv.DictReader(open(directory/'robust_centers.csv')))[0]
        promoted=list(csv.DictReader(open(directory/'inside_ball_promoted64.csv')))
        radius=float(ball['r_ball']) if ball['status']=='BALL_RESOLVED' else 0.0
        valid=(ball['status']=='BALL_RESOLVED' and int(center['successes'])>=63 and
               promoted and all(row['B63']=='True' for row in promoted) and radius>=0.20)
        if not valid:
            continue
        states.append(state_manifest); centers.append(center); balls.append(ball); inside.extend(promoted); eligible.append(state_manifest['state_id'])
    if not eligible:
        raise RuntimeError('no new margin-eligible anchors')
    (BALL/'frozen_state_manifest.json').write_text(json.dumps({'states':states},indent=2,sort_keys=True)+'\n')
    write_csv(BALL/'robust_centers.csv',centers)
    write_csv(BALL/'conservative_ball_parameters.csv',balls)
    write_csv(BALL/'inside_ball_promoted64.csv',inside)
    for name in ('ebridge_definition.json','ebridge_halfspaces.csv'):
        shutil.copy2(HERE/'anchor_builds'/eligible[0]/name,BALL/name)
    return eligible


def load_source():
    spec=importlib.util.spec_from_file_location('new_transfer_source',SOURCE)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    mod.OUT=OUT;mod.BALL=BALL;mod.CAP_CONT=12000;mod.CAP_STEPS=4000000
    def write_plan6(name,tasks):
        directory=OUT/'plans'/name;directory.mkdir(parents=True,exist_ok=True)
        ids=sorted({x['state_id'] for x in tasks})
        for shard in range(6):
            assigned={sid for index,sid in enumerate(ids) if index%6==shard}
            with (directory/f'shard{shard}.jsonl').open('w') as handle:
                for task in tasks:
                    if task['state_id'] in assigned:
                        handle.write(json.dumps(task,sort_keys=True)+'\n')
        summary={'name':name,'tasks':len(tasks),'shards':6,
                 'files':{f'shard{s}.jsonl':sha(directory/f'shard{s}.jsonl') for s in range(6)}}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    mod.write_plan=write_plan6
    return mod


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run-plan','advance','finalize'])
    parser.add_argument('--plan');parser.add_argument('--work')
    args=parser.parse_args()
    mod=load_source()
    if args.stage=='prepare':
        eligible=aggregate_eligible();mod.prepare();print(json.dumps({'eligible_new_anchors':eligible},indent=2))
    elif args.stage=='run-plan':
        mod.Runner(Path(args.plan),args.work).run()
    elif args.stage=='advance':mod.advance()
    else:mod.finalize()


if __name__=='__main__':
    main()
