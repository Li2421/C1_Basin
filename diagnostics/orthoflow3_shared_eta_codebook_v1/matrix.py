#!/usr/bin/env python3
"""Prepare, execute, and aggregate the frozen state-mode rollout matrix."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,sys,time,types
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1'
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
    fields=fields or (list(rows[0]) if rows else ['state_id'])
    with (H/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def prepare():
    states=json.load(open(H/'state_split.json'))['states'];cb=read(H/'codebook_eta.csv');tasks=[]
    nmap={'train':16,'val':32,'test':64}
    for st in states:
        n=nmap[st['split']]
        for m in cb:
            eta=[float(m[f'eta{i}']) for i in (1,2,3)]
            for fi in range(n):tasks.append(dict(state_id=st['state_id'],split=st['split'],controller=f"mode_{int(m['mode_id']):02d}",mode_id=int(m['mode_id']),eta=eta,future_index=fi,phase=f"{st['split']}_matrix"))
        if st['split'] in ('val','test'):
            for fi in range(n):tasks.append(dict(state_id=st['state_id'],split=st['split'],controller='safety',mode_id=-1,eta=[0.,0.,0.],future_index=fi,phase=f"{st['split']}_safety"))
    groups=defaultdict(list)
    for t in tasks:groups[(t['state_id'],t['controller'])].append(t)
    load=[0,0];owners={}
    for g in sorted(groups,key=lambda q:(-len(groups[q]),sha('|'.join(q)))):
        j=int(np.argmin(load));owners[g]=j;load[j]+=len(groups[g])
    p=H/'plans';p.mkdir(exist_ok=True)
    for j in range(2):
        with (p/f'shard{j}.jsonl').open('w') as f:
            for t in tasks:
                if owners[(t['state_id'],t['controller'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
    dump('rollout_cost_estimate.json',{'M':len(cb),'train':128*len(cb)*16,'val':32*len(cb)*32,'test':32*len(cb)*64,'safety_val':32*32,'safety_test':32*64,'total_continuations':len(tasks),'shard_tasks':load,'max_gpu_shards':2})
    print(json.dumps({'tasks':len(tasks),'shards':load}))
def prepare4():
    tasks=[]
    for p in sorted((H/'plans').glob('shard*.jsonl')):tasks += [json.loads(x) for x in open(p) if x.strip()]
    groups=defaultdict(list)
    for t in tasks:groups[(t['state_id'],t['controller'])].append(t)
    load=[0]*4;owners={}
    for g in sorted(groups,key=lambda q:(-len(groups[q]),sha('|'.join(q)))):
        j=int(np.argmin(load));owners[g]=j;load[j]+=len(groups[g])
    p=H/'plans4';p.mkdir(exist_ok=True)
    for j in range(4):
        with (p/f'shard{j}.jsonl').open('w') as f:
            for t in tasks:
                if owners[(t['state_id'],t['controller'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
    dump('rollout_cost_estimate_4shard.json',{'tasks':len(tasks),'shard_tasks':load,'prior_valid_cache_records':sum(sum(1 for _ in open(q)) for q in (H/'runs').glob('shard*/raw/pilot_rollouts.jsonl'))});print(json.dumps({'tasks':len(tasks),'shards':load}))
def prepare6():
    tasks=[]
    for p in sorted((H/'plans').glob('shard*.jsonl')):tasks += [json.loads(x) for x in open(p) if x.strip()]
    groups=defaultdict(list)
    for t in tasks:groups[(t['state_id'],t['controller'])].append(t)
    load=[0]*6;owners={}
    for g in sorted(groups,key=lambda q:(-len(groups[q]),sha('|'.join(q)))):
        j=int(np.argmin(load));owners[g]=j;load[j]+=len(groups[g])
    p=H/'plans6';p.mkdir(exist_ok=True)
    for j in range(6):
        with (p/f'shard{j}.jsonl').open('w') as f:
            for t in tasks:
                if owners[(t['state_id'],t['controller'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
    dump('rollout_cost_estimate_6shard.json',{'tasks':len(tasks),'shard_tasks':load,'prior_valid_cache_records':sum(sum(1 for _ in open(q)) for q in (H/'runs').glob('shard*/raw/pilot_rollouts.jsonl'))});print(json.dumps({'tasks':len(tasks),'shards':load}))
def run(shard):
    tasks=[json.loads(x) for x in open(H/f'plans/shard{shard}.jsonl') if x.strip()];allst=json.load(open(H/'state_split.json'))['states'];ids={t['state_id'] for t in tasks};states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in allst if x['state_id'] in ids};features=np.load(H/'state_features.npz')['features']
    # Local-only batch enlargement: four times the historical 64, while the
    # two-shard server limit stays fixed. The pilot cache makes restarts exact.
    text=(POINT/'run_eval_shard.py').read_text().replace("range(0,len(tasks),64)","range(0,len(tasks),256)").replace("chunk=tasks[begin:begin+64]","chunk=tasks[begin:begin+256]")
    mod=types.ModuleType('point_driver_local256');mod.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,mod.__file__,'exec'),mod.__dict__);out=H/f'runs/shard{shard}';out.mkdir(parents=True,exist_ok=True);m,O=mod.loadmod(out,states);m.CAP_CONT=1000000;m.CAP_STEPS=500000000;o=O(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,'codebook_matrix');dest=H/'raw';dest.mkdir(exist_ok=True)
    with open(dest/f'shard{shard}.jsonl','w') as f:
        for r in res:f.write(json.dumps({k:v for k,v in r.items() if k!='_source'},sort_keys=True)+'\n')
    dump(f'raw/shard{shard}_runtime.json',{'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st});print(json.dumps({'shard':shard,'tasks':len(tasks),'new':o.new,'steps':o.steps}))
def run4(shard):
    tasks=[json.loads(x) for x in open(H/f'plans4/shard{shard}.jsonl') if x.strip()];allst=json.load(open(H/'state_split.json'))['states'];ids={t['state_id'] for t in tasks};states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in allst if x['state_id'] in ids};features=np.load(H/'state_features.npz')['features']
    text=(POINT/'run_eval_shard.py').read_text().replace("range(0,len(tasks),64)","range(0,len(tasks),256)").replace("chunk=tasks[begin:begin+64]","chunk=tasks[begin:begin+256]");mod=types.ModuleType('point_driver_local256_4');mod.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,mod.__file__,'exec'),mod.__dict__);out=H/f'runs4/shard{shard}';out.mkdir(parents=True,exist_ok=True);m,O=mod.loadmod(out,states)
    class OracleReuse(O):
        def _load_prior(self):
            super()._load_prior()
            for p in sorted((H/'runs').glob('shard*/raw/pilot_rollouts.jsonl')):
                for line in open(p):
                    if line.strip():self._insert(json.loads(line),'valid_matrix_prior')
    m.CAP_CONT=1000000;m.CAP_STEPS=500000000;o=OracleReuse(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,'codebook_matrix');dest=H/'raw';dest.mkdir(exist_ok=True)
    with open(dest/f'shard{shard}.jsonl','w') as f:
        for r in res:f.write(json.dumps({k:v for k,v in r.items() if k!='_source'},sort_keys=True)+'\n')
    dump(f'raw/shard{shard}_runtime.json',{'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st,'reused_valid_two_shard_cache':True});print(json.dumps({'shard':shard,'tasks':len(tasks),'new':o.new,'steps':o.steps}))
def run6(shard):
    tasks=[json.loads(x) for x in open(H/f'plans6/shard{shard}.jsonl') if x.strip()];allst=json.load(open(H/'state_split.json'))['states'];ids={t['state_id'] for t in tasks};states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in allst if x['state_id'] in ids};features=np.load(H/'state_features.npz')['features']
    text=(POINT/'run_eval_shard.py').read_text().replace("range(0,len(tasks),64)","range(0,len(tasks),256)").replace("chunk=tasks[begin:begin+64]","chunk=tasks[begin:begin+256]");mod=types.ModuleType('point_driver_local256_6');mod.__file__=str(POINT/'run_eval_shard.py');exec(compile(text,mod.__file__,'exec'),mod.__dict__);out=H/f'runs6/shard{shard}';out.mkdir(parents=True,exist_ok=True);m,O=mod.loadmod(out,states)
    class OracleReuse(O):
        def _load_prior(self):
            super()._load_prior()
            for p in sorted((H/'runs').glob('shard*/raw/pilot_rollouts.jsonl')):
                for line in open(p):
                    if line.strip():self._insert(json.loads(line),'valid_matrix_prior')
    m.CAP_CONT=1000000;m.CAP_STEPS=500000000;o=OracleReuse(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,'codebook_matrix');dest=H/'raw';dest.mkdir(exist_ok=True)
    with open(dest/f'shard{shard}.jsonl','w') as f:
        for r in res:f.write(json.dumps({k:v for k,v in r.items() if k!='_source'},sort_keys=True)+'\n')
    dump(f'raw/shard{shard}_runtime.json',{'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-st,'reused_valid_two_shard_cache':True});print(json.dumps({'shard':shard,'tasks':len(tasks),'new':o.new,'steps':o.steps}))
def aggregate():
    rr=[]
    for p in sorted((H/'raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
    expect=json.load(open(H/'rollout_cost_estimate.json'))['total_continuations']
    if len(rr)!=expect:raise RuntimeError(('incomplete',len(rr),expect))
    states={x['state_id']:x for x in json.load(open(H/'state_split.json'))['states']};cb=read(H/'codebook_eta.csv');etas=np.array([[float(q[f'eta{i}']) for i in (1,2,3)] for q in cb])
    # Compatible point-search cache rows predate the aligned controller names.
    # Recover the frozen mode identity from their exact eta coordinate.
    for r in rr:
        if 'controller' not in r:
            e=np.asarray(r['eta'],float)
            if np.linalg.norm(e)<=1e-12:r['controller']='safety';r['mode_id']=-1
            else:
                j=int(np.argmin(np.linalg.norm(etas-e,axis=1)))
                if np.linalg.norm(etas[j]-e)>1e-12:raise RuntimeError(('unmapped cached eta',e))
                r['controller']=f'mode_{j:02d}';r['mode_id']=j
            r['split']=states[r['state_id']]['split']
    by=defaultdict(list)
    for r in rr:by[(r['state_id'],r['controller'])].append(r)
    rows={'train':[],'val':[],'test':[]};safety={'val':[],'test':[]}
    for (sid,ctl),v in sorted(by.items()):
        out=Counter(x['outcome'] for x in v);k=sum(x['success'] for x in v);js=[x['J_def'] for x in v if x['success']];base=dict(state_id=sid,split=states[sid]['split'],controller=ctl,mode_id=-1 if ctl=='safety' else int(ctl.split('_')[1]),successes=k,trials=len(v),empirical_Q=k/len(v),B63=(len(v)==64 and k>=63),deadlock=out['safe_deadlock'],timeout=out['timeout'],collision=out['collision'],J_def_mean=float(np.mean(js)) if js else '',episode_length_mean=float(np.mean([x['continuation_steps'] for x in v])))
        if ctl=='safety':safety[states[sid]['split']].append(base)
        else:rows[states[sid]['split']].append(base)
    write('train_mode_counts.csv',rows['train']);write('val_mode_counts.csv',rows['val']);write('test_mode_q64.csv',rows['test']);write('safety_val_q32.csv',safety['val']);write('safety_test_q64.csv',safety['test'])
    dump('matrix_summary.json',{'rows':{s:len(rows[s]) for s in rows},'safety':{s:len(safety[s]) for s in safety},'continuations':len(rr),'collisions':sum(r['collision'] for s in rows for r in rows[s])+sum(r['collision'] for s in safety for r in safety[s])})
    w=json.load(open(H/'working_state.json'));w.update(status='MODE_MATRIX_COMPLETE',completed=w['completed']+['state_mode_rollouts'],next_action='train baselines and MLP');dump('working_state.json',w);print(json.dumps(json.load(open(H/'matrix_summary.json')),indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','prepare4','prepare6','run','run4','run6','aggregate']);ap.add_argument('--shard',type=int);a=ap.parse_args();{'prepare':prepare,'prepare4':prepare4,'prepare6':prepare6,'run':lambda:run(a.shard),'run4':lambda:run4(a.shard),'run6':lambda:run6(a.shard),'aggregate':aggregate}[a.stage]()
if __name__=='__main__':main()
