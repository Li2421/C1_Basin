"""Build critic data: train scenes use states 0-15 labels; validation scenes give inputs only (labels kept separately)."""
import json,pickle,numpy as np,desc
from tab import *
sp=json.load(open('SPLIT.json'));S={s['name']:s for s in json.load(open('scenes_set.json'))}
T=load(['design.jsonl','extra.jsonl','fresh.jsonl'])
data={}
for n in sp['train']+sp['val']:
 s=dict(S[n])
 if s['cluster']=='obst':s['min_opening_over_diameter']=desc.literature_opening('lit_swap8_obstacles',s['walls'],s['config']['agent_radius'])
 s['features']=desc.features(s);c=desc.descriptor(s);h=desc.state_h(s)
 t=T[n].copy()
 if n in sp['train']:t[:,16:]=np.nan
 else:t[:]=np.nan   # validation labels never enter training data
 data[n]=dict(t=t,c=c,h=h,h_pred=h)
pickle.dump(data,open('critic_data.pkl','wb'))
print({n:(np.isfinite(d['t']).sum(),np.round(d['c'],2).tolist()) for n,d in list(data.items())[:3]})
