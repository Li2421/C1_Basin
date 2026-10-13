"""Exact-semantic FEW_x2 precision supplement; no new environment identity."""
from pathlib import Path
import json,copy,argparse,time
from shared_rollout_db.src.rollout_db import connect,transaction,canonical,uid
from shared_rollout_db.src.cache_writer import append_journal
from diagnostics.controller_state_transfer_v1.study import read,write,freeze,sha
HERE=Path(__file__).resolve().parent/'legacy_supplement';SOURCE=HERE.parents[1]/'controller_state_transfer_v1';EXPERIMENT='exp_controller_state_geometry_transfer_v1_legacy_supplement'

def design():
 cases=[]
 for old in read(SOURCE/'cases.json'):
  if old['width'] in [.41,.59] and 8<=old['state_index']<24 and old['seed']==0:
   for sd in [2,3]:
    c=copy.deepcopy(old);sk=json.loads(c['seed_key']);sk['future_index']=sd;c['seed']=sd;c['seed_key']=canonical(sk);c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));cases.append(c)
 freeze(HERE/'cases.json',cases);freeze(HERE/'protocol.json',{'source_protocol_sha256':sha(SOURCE/'protocol.json'),'source_runtime_sha256':sha(SOURCE/'runtime.py'),'preregistration_sha256':sha(HERE.parent/'ADDENDUM_DIAGNOSTICS_AND_FACTORS.md'),'n_cases':len(cases)})
 freeze(HERE/'planned_rollouts.json',{'requests':[dict(state_uid=c['state_uid'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[c['seed_key']]) for c in cases]})
 with connect() as db,transaction(db):db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXPERIMENT,'Legacy Toy precision supplement',str(HERE),sha(HERE/'protocol.json'),sha(SOURCE/'runtime.py'),canonical({'new_labels':1024})))
 print(len(cases))

def worker(index,workers):
 from diagnostics.controller_state_transfer_v1.runtime import base_runtime,controller,rollout
 import jax
 jax.config.update('jax_enable_x64',True)
 p=read(HERE/'protocol.json');assert sha(SOURCE/'runtime.py')==p['source_runtime_sha256']
 archive=read(SOURCE.parent/'controller_ff_factorial_repair_v1/protocol.json');states={s['state_uid']:s for s in archive['states']};pf=read(HERE/'cache_preflight.json');assert pf['summary']['ambiguous']==0
 missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],k) for r in pf['details'] for k in r['missing_seeds']};rts={};buf=[];t=time.monotonic();count=0
 for c in read(HERE/'cases.json')[index::workers]:
  dest=HERE/'raw'/(c['rollout_uid']+'.json')
  if dest.exists() or (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) not in missing:continue
  key=(c['width'],c['controller_name'])
  if key not in rts:rts[key]=controller(base_runtime(c['width']),c['controller_name'])
  r=rollout(rts[key],states[c['parent_state_uid']],c['eta'],c['seed']);row=dict(schema='csg_supplement_seed_v1',experiment_uid=EXPERIMENT,case=c,protocol_sha256=sha(HERE/'protocol.json'),execution_code_sha256=sha(SOURCE/'runtime.py'),**r)
  write(dest,row);buf.append(row);count+=1
  if len(buf)>=8:append_journal(buf,EXPERIMENT,str(index));buf=[]
 if buf:append_journal(buf,EXPERIMENT,str(index))
 write(HERE/'workers'/f'{index}.json',dict(executed=count,seconds=time.monotonic()-t,completed=True))

def merge():
 from . import revision_study as s
 s.HERE=HERE;s.EXPERIMENT=EXPERIMENT
 # Standard merger accepts main_* journals.
 from shared_rollout_db.src.rollout_db import ROOT
 # Use original merge algorithm with a broadened journal glob, without touching history.
 import inspect
 code=inspect.getsource(s.merge).replace("glob('main_*.jsonl')","glob('*.jsonl')")
 ns=dict(s.__dict__);exec(code,ns);ns['merge']()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['design','worker','merge']);a.add_argument('--index',type=int,default=0);a.add_argument('--workers',type=int,default=16);x=a.parse_args();{'design':design,'merge':merge}.get(x.action,lambda:worker(x.index,x.workers))()
