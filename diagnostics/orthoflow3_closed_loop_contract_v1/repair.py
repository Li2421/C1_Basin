"""Isolated minimal repair: expose remaining horizon, unchanged physical labels."""
import argparse,json,shutil,os
_backend=os.environ.get('JAX_PLATFORMS','cpu')
_devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b
os.environ['JAX_PLATFORMS']=_backend
os.environ['CUDA_VISIBLE_DEVICES']=_devices

OUT=a.ROOT/'diagnostics/orthoflow3_closed_loop_contract_v1/repair'
DATA=a.ROOT/'datasets/orthoflow3_basin_dataset_v2_contract'
OLD_OUT=b.OUT;OLD_DATA=b.DATA
ORIGINAL_TRANSFORM=b.transformed
VERSION='phase_B_plus_remaining_horizon_v1'

def remaining(row):
    p=a.decode(row['structured_state'])
    if 'normalized_episode_time' in p:t=float(p['normalized_episode_time'])
    elif int(p['timestep'])==0:t=0.
    else:raise ValueError('Missing authoritative horizon metadata')
    if not 0<=t<=1:raise ValueError(t)
    return 1.-t

def append(row):
    c=a.decode(row['conditioning'])
    return {**row,'conditioning':json.dumps({**c,'flat':[*c['flat'],remaining(row)],'schema':VERSION}),
            'conditioning_version':VERSION}

def transformed(row):return append(ORIGINAL_TRANSFORM(row))

def build():
    if (DATA/'manifest.json').exists():return a.load(DATA/'manifest.json')
    proof=a.load(OUT.parent/'waiting_probe/results.json')
    witness=a.load(OUT.parent/'waiting_probe/preregistration.json')['witnesses']
    by={(x['state_uid'],x['eta_index']):x for x in proof}
    assert all(w['h_max_error']==0 for w in witness)
    assert any(by[(w['state_pair'][0],1)]['successes']>=12 and by[(w['state_pair'][1],1)]['successes']==0 for w in witness)
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    rr=[append(r) for r in pq.read_table(OLD_DATA/'states.parquet').to_pylist()]
    pq.write_table(pa.Table.from_pylist(rr),DATA/'states.parquet');shutil.copyfile(OLD_DATA/'eta_labels.parquet',DATA/'eta_labels.parquet')
    extra=[append(r) for r in a.load(OLD_OUT/'initial_expansion_canonical.json')]
    (OUT/'initial_expansion_canonical.json').write_text(json.dumps(extra))
    norm=a.load(OLD_OUT/'normalization.json')
    for n in norm['scenarios'].values():n['h_mean'].append(0.);n['h_std'].append(1.)
    norm['remaining_horizon_normalization']='physical fraction, no fitting'
    (OUT/'normalization.json').write_text(json.dumps(norm,indent=2))
    m={'source':str(OLD_DATA),'version':VERSION,'only_change':'append 1-normalized_episode_time',
       'same_state_ids':True,'same_labels_sha256':a.sha(DATA/'eta_labels.parquet'),
       'source_states_sha256':a.sha(OLD_DATA/'states.parquet'),'states_sha256':a.sha(DATA/'states.parquet'),
       'new_physical_labels_for_reencoding':0,'train_validation_split_unchanged':True,
       'architecture_loss_budget':'identical Phase B','seeds':list(b.base.SEEDS)}
    (DATA/'manifest.json').write_text(json.dumps(m,indent=2));return m

def install():
    b.OUT=OUT;b.DATA=DATA;b.transformed=transformed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','generator','critic','freeze_critic']);p.add_argument('--seed',type=int,default=17);x=p.parse_args()
    install()
    print(json.dumps({'build':build,'generator':b.train_generator,'critic':lambda:b.train_critic(x.seed),'freeze_critic':b.freeze_critic}[x.stage](),indent=2))
