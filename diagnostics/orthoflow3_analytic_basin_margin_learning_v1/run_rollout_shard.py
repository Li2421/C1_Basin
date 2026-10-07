#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_analytic_basin_margin_learning_v1';POINT=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1';OLD=ROOT/'diagnostics/orthoflow3_conservative_basin_ball_pilot6_v1/pilot6.py';FUTURE=2026092811
def loadmod(out,states):
 text=OLD.read_text().replace('for begin in range(0,len(tasks),32):','for begin in range(0,len(tasks),64):').replace('chunk=tasks[begin:begin+32]','chunk=tasks[begin:begin+64]')
 oa='                actions=np.asarray(self.sample(jnp.asarray(obs),jnp.asarray(np.stack(keys))))';na="""\
                at_query_time=all(int(envs[i].step_count)==int(self.states[chunk[i]['state_id']]['absolute_step']) for i in active)
                if at_query_time:
                    actions=np.concatenate([np.asarray(self.sample(jnp.asarray(obs[j:j+1]),jnp.asarray(np.stack(keys[j:j+1])))) for j in range(len(active))],axis=0)
                else:
                    actions=np.asarray(self.sample(jnp.asarray(obs),jnp.asarray(np.stack(keys))))""";assert oa in text;text=text.replace(oa,na)
 spec=importlib.util.spec_from_loader('analytic_rollout_old',loader=None);m=importlib.util.module_from_spec(spec);m.__file__=str(OLD);sys.modules[spec.name]=m;exec(compile(text,m.__file__,'exec'),m.__dict__);m.HERE=out;m.FUTURE_ROOT=FUTURE;m.CAP_CONT=20000;m.CAP_STEPS=10000000
 class Oracle(m.Oracle):
  def _load_prior(self):
   if self.record_path.exists():
    for line in self.record_path.read_text().splitlines():
     if line.strip():r=json.loads(line);self.rows.append(r);self._insert(r,'pilot')
   self.new=len(self.rows);self.steps=sum(int(r['continuation_steps']) for r in self.rows)
 return m,Oracle
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();tasks=[json.loads(x) for x in open(HERE/'plans'/a.plan/f'shard{a.shard}.jsonl') if x.strip()]
 allstates=json.load(open(POINT/'final_source_split.json'))['states'];needed={t['state_id'] for t in tasks};states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in allstates if x['state_id'] in needed};features=np.load(POINT/'point_learning_arrays.npz')['features']
 out=HERE/'runs'/a.plan/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True);m,O=loadmod(out,states);o=O(states,features,np.zeros(3),np.ones(3));start=time.time();result=o.ensure(tasks,a.plan)
 tm={(t['state_id'],tuple(float(z) for z in t['eta']),int(t['future_index'])):t for t in tasks};dest=HERE/'raw'/a.plan;dest.mkdir(parents=True,exist_ok=True)
 with open(dest/f'shard{a.shard}.jsonl','w') as f:
  for row in result:
   k=(row['state_id'],tuple(float(z) for z in row['eta']),int(row['future_index']));z=dict(tm[k]);z.update({kk:vv for kk,vv in row.items() if kk!='_source'});f.write(json.dumps(z,sort_keys=True)+'\n')
 rt={'plan':a.plan,'shard':a.shard,'tasks':len(tasks),'new_continuations':o.new,'physical_steps':o.steps,'wall_seconds':time.time()-start};(dest/f'shard{a.shard}_runtime.json').write_text(json.dumps(rt,indent=2)+'\n');print(json.dumps(rt))
if __name__=='__main__':main()
