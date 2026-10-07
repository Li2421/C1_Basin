"""Parallel scheduling wrapper only; immutable controller runtime unchanged."""
import argparse,copy
from pathlib import Path
from . import middle_phase_alias as runtime


def main(index):
    original=runtime.read;doc=copy.deepcopy(original(runtime.OUT/'cache_preflight.json'));counter=0
    for row in doc['details']:
        keep=[]
        for sk in row['missing_seeds']:
            if counter%3==index//2:keep.append(sk)
            counter+=1
        row['missing_seeds']=keep
    runtime.read=lambda path:doc if Path(path)==runtime.OUT/'cache_preflight.json' else original(path)
    runtime.run(index%2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();main(a.index)
