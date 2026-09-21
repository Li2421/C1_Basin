"""Train the unchanged Stage-I Flow-BC architecture on validated SI data only."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.validate import validate_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/give_way_si_short_v1')
    p.add_argument('--out_dir',type=Path,default=ROOT/'single_integrator/runs/short_uniform_state_normalized_seed0')
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--normalize',action='store_true',default=True)
    p.add_argument('--training_data_dir',type=Path,default=ROOT/'datasets/give_way_si_short_uniform_state_v1/raw',help='Audited one-to-one independent state augmentation of this dataset')
    args=p.parse_args()
    if min(args.steps,args.batch_size)<=0:raise ValueError('steps and batch_size must be positive')
    metadata,validation=validate_dataset(args.dataset)
    print('Validated SI data:',validation,flush=True)
    sources=['single_integrator/environment.py','single_integrator/expert.py','single_integrator/generate.py','flowbc/train.py','flowbc/giveway_dataset.py','flowbc/giveway_flowbc_agent.py']
    provenance=dict(method='stage_i_joint_flow_bc',not_full_mac=True,dataset=str(args.dataset.resolve()),dataset_sha256=validation['dataset_sha256'],evaluation_environment=metadata['evaluation_environment'],training_environment=metadata['environment'],source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources})
    provenance.update(normalize=args.normalize,sampling='uniform transitions; no stage weighting',training_seed=args.seed)
    training_data=args.dataset/'raw'
    if args.training_data_dir:
        import numpy as np
        training_data=args.training_data_dir
        aug=json.loads((training_data.parent/'environment.json').read_text())
        if aug.get('schema')!='giveway_si_uniform_state_v1' or not aug.get('complete') or aug['source_dataset_sha256']!=validation['dataset_sha256']:
            raise ValueError('Unverified augmentation or mismatched source')
        digest=hashlib.sha256()
        paths=sorted(training_data.glob('*.npz'))
        if {f.name for f in paths}!={f'episode_{i:04d}.npz' for i in range(450)}:raise ValueError('Expected precisely train/val pairs')
        for path in paths:
            digest.update(path.name.encode());digest.update(hashlib.sha256(path.read_bytes()).digest())
            with np.load(path) as z,np.load(args.dataset/'raw'/path.name) as original:
                np.testing.assert_array_equal(z['source_step'],np.arange(len(original['actions'])))
                np.testing.assert_array_equal(z['stages'],original['stages'])
                if int(z['pair_id'])!=int(original['pair_id']) or int(z['mode'])!=int(original['mode']):raise ValueError('Pair/mode changed')
                if z['wall_collision'].any() or z['agent_collision'].any():raise ValueError('Unsafe training label')
                np.testing.assert_allclose(z['next_observations'][:,:,:2],z['observations'][:,:,:2]+metadata['environment']['dt']*z['actions'],atol=5e-7,rtol=0)
        if digest.hexdigest()!=aug['dataset_sha256']:raise ValueError('Augmentation digest mismatch')
        provenance.update(training_data_augmentation=aug,training_data_dir=str(training_data.resolve()))
    subprocess.run([sys.executable,'-u',str(ROOT/'flowbc/train.py'),'--data_dir',str(training_data),'--out_dir',str(args.out_dir),'--train_steps',str(args.steps),'--seed',str(args.seed),'--batch_size',str(args.batch_size),'--log_interval',str(min(10000,args.steps)),'--save_interval',str(min(25000,args.steps)),*(['--normalize'] if args.normalize else [])],check=True)
    for path in args.out_dir.glob('*.pkl'):
        with path.open('rb') as f:checkpoint=pickle.load(f)
        checkpoint['si_metadata']=provenance
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:pickle.dump(checkpoint,f)
        temporary.replace(path)
    (args.out_dir/'benchmark.json').write_text(json.dumps(provenance,indent=2))
    print('SI provenance attached; method is Stage-I Flow-BC, not full MAC.',flush=True)


if __name__=='__main__':main()
