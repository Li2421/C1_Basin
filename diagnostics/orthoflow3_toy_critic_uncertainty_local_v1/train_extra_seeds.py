#!/usr/bin/env python3
"""Add two independent W1 pure-NLL critics; preserve prior architecture/data/split."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
from flax import serialization
ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=Path(__file__).resolve().parent
SRC=ROOT/'diagnostics/orthoflow3_nll_weighting_ablation_v1/run_experiment.py'
spec=importlib.util.spec_from_file_location('frozen_w1_training',SRC)
lib=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=lib
spec.loader.exec_module(lib)

def main():
    sr,wr,cr,vr,_=lib.prepare()
    structured,wide,combined,val=map(lib.lib.normalize,(sr,wr,cr,vr))
    assert len(sr)==1920 and len(cr)==2876 and len(vr)==144
    for seed in (47,59):
        folder=HERE/f'seed{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        cp=folder/'checkpoint.msgpack'
        if cp.exists():
            print(json.dumps({'seed':seed,'reused_checkpoint':str(cp)}),flush=True)
            continue
        started=time.monotonic()
        _,best,_traj,_tw=lib.train_one('secondary_combined','W1',seed,structured,wide,combined,val)
        cp.write_bytes(serialization.to_bytes(best[1]))
        summary={'seed':seed,'dataset':'secondary_combined','weighting':'W1_PAIR_EQUAL',
                 'architecture':'SingleCritic','objective':'pure_NLL','train_pairs':len(cr),
                 'val_pairs':len(vr),'best_val_objective':best[0],'best_step':best[2],
                 'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
                 'wall_seconds':time.monotonic()-started}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
