"""Frozen Stage B checkpoints on wholly new Stage C states; no labels read."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent
HERE=BASE/'stage_c'

import jax
import numpy as np
from flax import serialization
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
from diagnostics.controller_loco_real_flow_v1.train import Net,descriptor,ARMS

def read(p):return json.loads(Path(p).read_text())

def main(fold,arm):
    assert arm in ARMS and 0<=fold<7
    p=read(HERE/'protocol.json');assert fold in p['held_folds']
    jax.config.update('jax_enable_x64',False)
    assert any(d.platform=='gpu' for d in jax.devices())
    names=p['controller_order'];phys=physical23(p['states']);state_uids=[s['state_uid'] for s in p['states']]
    def feats(name):
        z=np.load(HERE/'static_features'/f'{name}.npz')
        assert list(z['state_uids'])==state_uids
        return np.concatenate((phys,descriptor(z,arm)),axis=-1).astype(np.float32)
    correct=feats(names[fold]);wrong=np.stack([feats(names[c]) for c in range(7) if c!=fold])
    base=BASE/'stage_b/fits'/f'fold{fold}_{arm}';meta=read(base/'complete.json');norm=np.load(base/'normalization.npz')
    h=(correct-norm['mean'])/norm['scale'];w=(wrong-norm['mean'])/norm['scale']
    net=Net();template=net.init(jax.random.PRNGKey(0),np.zeros((1,343),np.float32))
    held=[];mismatched=[]
    for path in meta['checkpoints']:
        params=serialization.from_bytes(template,Path(path).read_bytes())
        held.append(np.asarray(net.apply(params,h)))
        mismatched.append(np.asarray(net.apply(params,w.reshape(-1,343))).reshape(6,64,16))
    dest=HERE/'predictions'/f'fold{fold}_{arm}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,held=np.asarray(held),mismatched=np.asarray(mismatched),state_uids=np.array(state_uids))
    receipt=dict(fold=fold,arm=arm,source_checkpoint_files=meta['checkpoints'],frozen_step=meta['selected_step'],
        source_normalization=str(base/'normalization.npz'),jax_enable_x64=False,test_labels_read=False,states=64,
        physical_steps=0)
    (HERE/'predictions'/f'fold{fold}_{arm}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
