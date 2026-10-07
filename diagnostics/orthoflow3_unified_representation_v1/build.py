import json,shutil,hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import representation as rep

ROOT=rep.ROOT;OUT=Path(__file__).resolve().parent;DATA=ROOT/'datasets/orthoflow3_basin_dataset_v3_unified_rep'
SOURCE=ROOT/'datasets/orthoflow3_basin_dataset_v2_audited'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def rows():return pq.read_table(DATA/'states.parquet').to_pylist()
def scene(row):return rep.decode(row['conditioning'])['physical_entities']

def convert(row):
    s=rep.parse(row);e=rep.entities(s)
    return {**row,'native_conditioning':row['conditioning'] if isinstance(row['conditioning'],str) else json.dumps(row['conditioning']),
            'conditioning':json.dumps({'schema':rep.VERSION,'physical_entities':s}),
            'num_agents':len(s['positions']),'num_obstacles':len(s['obstacles']),
            'conditioning_version':rep.VERSION,'source_state_uid':row['state_uid']}

def build():
    assert json.loads((ROOT/'diagnostics/orthoflow3_closed_loop_contract_v1/decision.json').read_text())['status']=='CONTRACT_REPAIRED_AND_VALIDATED'
    if (DATA/'manifest.json').exists():return json.loads((DATA/'manifest.json').read_text())
    source=pq.read_table(SOURCE/'states.parquet').to_pylist();rr=[convert(r) for r in source]
    DATA.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rr),DATA/'states.parquet')
    shutil.copyfile(SOURCE/'eta_labels.parquet',DATA/'eta_labels.parquet')
    extra=json.loads((ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/initial_expansion/states.json').read_text())
    ee=[convert(r) for r in extra];dump('initial_expansion_entities.json',ee)
    for split in ('train','validation'):
        (DATA/f'split_{split}.json').write_text(json.dumps([{'state_uid':r['state_uid'],'parent_episode_id':r['parent_episode_id'],'scenario':r['scenario']} for r in rr if r['split']==split],indent=2))
    assert {r['state_uid'] for r in source}=={r['state_uid'] for r in rr}
    parents={s:{r['parent_episode_id'] for r in rr+ee if r['split']==s} for s in ('train','validation')}
    assert not parents['train']&parents['validation']
    m={'version':rep.VERSION,'source':str(SOURCE),'state_count':len(rr),'additional_existing_critic_states':len(ee),
       'states_sha256':sha(DATA/'states.parquet'),'source_states_sha256':sha(SOURCE/'states.parquet'),
       'labels_sha256':sha(DATA/'eta_labels.parquet'),'labels_byte_identical':sha(DATA/'eta_labels.parquet')==sha(SOURCE/'eta_labels.parquet'),
       'encoder_source_sha256':sha(rep.__file__),'normalization':'fixed physical feature semantics; no fitted stats',
       'new_rollouts_for_reencoding':0,'parent_split_leakage':0,'scenario_id_learned':False,
       'counts':{sc:{split:sum(r['scenario']==sc and r['split']==split for r in rr) for split in ('train','validation')} for sc in sorted({r['scenario'] for r in rr})}}
    (DATA/'manifest.json').write_text(json.dumps(m,indent=2));dump('dataset_manifest.json',m);return m

if __name__=='__main__':print(json.dumps(build(),indent=2))
