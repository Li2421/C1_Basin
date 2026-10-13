"""Frozen controller-level η prior using only initial-state static queries."""
import os
import sys
import json
import hashlib
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent
import jax
import numpy as np
from flax import serialization
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
from diagnostics.controller_loco_real_flow_v1.train import Net,descriptor

def main(index):
    tasks=json.loads((BASE/'controller_prior/PREDICTION_TASKS.json').read_text())
    stage,fold,arm=tasks[index]
    p=json.loads((BASE/'protocol.json').read_text());name=p['controller_order'][fold]
    z=np.load(BASE/'static_features'/f'{name}.npz')
    assert list(z['state_uids'][:128])==[s['state_uid'] for s in p['states'][:128]]
    x=np.concatenate((physical23(p['states'][:128]),descriptor(z,arm)[:128]),axis=-1).astype(np.float32)
    fit=(BASE/'fits' if stage=='stage_a' else BASE/'stage_b/fits')/f'fold{fold}_{arm}'
    meta=json.loads((fit/'complete.json').read_text());norm=np.load(fit/'normalization.npz')
    assert name not in meta['source_controller_names']
    assert meta['jax_enable_x64'] is False
    jax.config.update('jax_enable_x64',False)
    assert any(d.platform=='gpu' for d in jax.devices())
    x=(x-norm['mean'])/norm['scale']
    net=Net();template=net.init(jax.random.PRNGKey(0),np.zeros((1,343),np.float32))
    logits=[];hashes=[]
    for filename in meta['checkpoints']:
        blob=Path(filename).read_bytes();params=serialization.from_bytes(template,blob)
        logits.append(np.asarray(net.apply(params,x)));hashes.append(hashlib.sha256(blob).hexdigest())
    logits=np.asarray(logits);prior=np.asarray(jax.nn.sigmoid(logits)).mean(1).argmax(-1)
    dest=BASE/'controller_prior'/f'{stage}_fold{fold}_{arm}'
    np.savez_compressed(dest.with_suffix('.npz'),logits=logits,eta_prior=prior,state_uids=z['state_uids'][:128])
    receipt=dict(stage=stage,fold=fold,arm=arm,eta_prior=prior.tolist(),fit_seeds=meta['fit_seeds'],
                 state_count=128,target_labels_read=False,test_labels_read=False,
                 checkpoint_sha256=hashes,checkpoints=meta['checkpoints'],jax_enable_x64=False)
    dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main(int(sys.argv[1]))
