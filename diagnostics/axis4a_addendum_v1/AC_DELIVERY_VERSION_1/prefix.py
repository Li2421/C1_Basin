"""Causal prefix compatibility; no physical simulation or inferred future labels."""
from core import *
from progress import progress

def eta(c,t):return c['eta_a'] if t<c['switch_step'] else c['eta_b']
def first_difference(c,old,positions,goals,source_outcome):
 n=len(positions)-1;H=c['H'];limit=min(H,n if source_outcome in ['success','collision'] else n+1)
 if c['arm']=='nominal_only' or old['arm']=='nominal_only':
  assert c['arm']==old['arm'],'Nominal/safety are not equivalent';return None
 if c['arm']=='Sprog':
  assert old['arm']=='F13';d0=np.linalg.norm(goals-positions[0],axis=-1);p=np.clip(1-np.linalg.norm(goals[None]-positions,axis=-1)/d0[None],0,1).mean(1);hit=np.flatnonzero(p[:limit]>=c['progress_threshold']);candidate=min(int(hit[0]) if len(hit) else H,round(6*H/7));return candidate if candidate<limit else None
 points=sorted(set([0,old['switch_step'],c['switch_step']]))
 for t in points:
  if t<limit and np.asarray(eta(c,t),np.float64).tobytes()!=np.asarray(eta(old,t),np.float64).tobytes():return t
 return None

def choose_prefix(c,old,positions,goals,source_outcome):
 n=len(positions)-1;diff=first_difference(c,old,positions,goals,source_outcome);end=min(n,c['H'],diff if diff is not None else c['H']);kind='CONTINUE';term=None
 if source_outcome in ['success','collision'] and n<=c['H'] and (diff is None or diff>=n):kind='TERMINAL_REUSE';term=source_outcome;end=n
 elif c['H']<=n and (diff is None or diff>=c['H']):kind='BUDGET_PREFIX';term='timeout';end=c['H']
 elif source_outcome=='numerical_failure' and n<c['H'] and diff is None:kind='UNKNOWN_REUSE';term='numerical_failure';end=n
 return dict(steps=end,kind=kind,termination=term,first_action_difference=diff)
