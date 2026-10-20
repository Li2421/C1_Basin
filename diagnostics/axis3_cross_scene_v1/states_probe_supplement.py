from common import *
from states import signature
from registry import normalized_state
from shared_rollout_db.src.rollout_db import canonical,uid

def main():
 states=read(HERE/'STATES.json');sc='four_way_intersection'
 if sc not in states or 'probe' in states[sc]:return
 from four_way_intersection.environment import Config,FourWayIntersectionEnv
 from four_way_intersection.scenario import sample_initial_state,build_layout
 inv=read(HERE/'HISTORICAL_INVENTORY.json');cfg=read(HERE/'DESIGN_CONFIG.json');cc=Config(**inv['environments'][sc]['config']['config']);sid=inv['environments'][sc]['scenario_uid'];out=[]
 for i in range(16):
  seed=cfg['probe_start']+cfg['scene_offsets'][sc]+i;p,v,m=sample_initial_state(cc,'development',i,base_seed=cfg['probe_start']+cfg['scene_offsets'][sc]);g=build_layout(cc).goals;FourWayIntersectionEnv(cc).reset(p,velocities=v);physical=clean(dict(positions=p,velocities=v,goals=g,timestep=0));alias=f'round34_{sc}_probe_{i}_{seed}';content=uid('content',dict(physical=physical,reset='native true t0'));s=dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),scenario=sc,scenario_uid=sid,content_hash=content,physical=physical,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),family=alias,study_index=i,study_split='probe',metadata=dict(draw_seed=seed,split='probe',outcome_blind=True));out.append(s)
 with rollout_db.connect(True) as db:old=[dict(r) for r in db.execute('SELECT physical_state_json FROM state')]
 sigs=set()
 for r in old:
  try:h=signature(json.loads(r['physical_state_json']))
  except (ValueError,TypeError):h=None
  if h:sigs.add(h)
 for splits in states.values():
  for rows in splits.values():sigs.update(signature(s['physical']) for s in rows)
 hits=[]
 for s in out:
  h=signature(s['physical'])
  if h in sigs:hits.append(s['uid'])
  sigs.add(h)
 write(HERE/'FOUR_PROBE_DEDUP.json',dict(utc=utc(),passed=not hits,states=16,duplicates=hits));assert not hits
 write(HERE/'STATES_BEFORE_FOUR_PROBE_SUPPLEMENT.json',states);states[sc]['probe']=[normalized_state(s) for s in out];write(HERE/'STATES.json',states);seal('FOUR_PROBE_STATE_SUPPLEMENT.receipt.json',[HERE/'STATES.json',HERE/'STATES_BEFORE_FOUR_PROBE_SUPPLEMENT.json',HERE/'FOUR_PROBE_DEDUP.json',HERE/'PROBE_DURATION_AMENDMENT.receipt.json'],pilot_eval_states_unchanged=True)
if __name__=='__main__':main()
