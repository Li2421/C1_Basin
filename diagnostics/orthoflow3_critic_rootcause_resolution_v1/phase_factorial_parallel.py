"""Scheduling-only partition of immutable goal-phase runtime; no physics edits."""
import argparse,copy
from pathlib import Path
from .phase_factorial_support import OUT,NAMES,read,runtime,sha


def main(index):
    name=NAMES[index//6];partition=(index%6)//2;shard=index%2;dest=OUT/name
    assert sha(runtime.__file__)==read(dest/'protocol.json')['intervention_runtime_sha256']
    doc=copy.deepcopy(read(dest/'execution_preflight.json'));counter=0
    for row in doc['details']:
        keep=[]
        for sk in row['missing_seeds']:
            if counter%3==partition:keep.append(sk)
            counter+=1
        row['missing_seeds']=keep
    runtime.OUT=dest;runtime.EXP=read(dest/'protocol.json')['experiment_uid'];original=runtime.read
    runtime.read=lambda path:doc if Path(path)==dest/'cache_preflight.json' else original(path)
    runtime.run(shard)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();main(a.index)
