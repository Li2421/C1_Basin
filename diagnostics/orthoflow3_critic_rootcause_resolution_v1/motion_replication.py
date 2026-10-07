"""Prospective replication with unchanged source models and execution code."""
import argparse,shutil
from pathlib import Path
from . import motion_confirmation as run
from . import motion_confirmation_eval as evaluate
from . import motion_confirmation_authorize as authorize

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'motion_replication_models'
RULE=ROOT/'motion_replication_protocol.json'
read,write,sha=run.read,run.write,run.sha


def configure():
    SOURCE.mkdir(exist_ok=True)
    origin=ROOT/'motion_final_source_models/models_frozen.json'
    destination=SOURCE/'models_frozen.json'
    if not destination.exists():shutil.copyfile(origin,destination)
    assert sha(origin)==sha(destination)
    guard=dict(wrapper_sha256=sha(__file__),rule_sha256=sha(RULE),models_sha256=sha(origin),
        controller_code_sha256=sha(run.__file__),evaluator_code_sha256=sha(evaluate.__file__),
        target_labels_used_for_models=False)
    if (SOURCE/'wrapper_frozen.json').exists():assert read(SOURCE/'wrapper_frozen.json')==guard
    else:write(SOURCE/'wrapper_frozen.json',guard)
    run.SOURCE,evaluate.SOURCE,authorize.SOURCE=SOURCE,SOURCE,SOURCE
    run.RULE,evaluate.RULE=RULE,RULE
    def folder(replica):
        rule=read(RULE)['target_rule'][replica]
        return ROOT/f'motion_independent_confirmation_{rule["controller_seed"]}'
    evaluate.folder,authorize.folder=folder,folder


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action')
    p.add_argument('--replicate',type=int,default=0);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();configure()
    if a.action=='predict':evaluate.predict(a.replicate)
    elif a.action=='evaluate':evaluate.evaluate(a.replicate)
    elif a.action=='authorize':authorize.main(a.replicate)
    else:run.main(a.action,a.replicate,a.index)
