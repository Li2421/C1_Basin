"""Pure before-action progress/trigger decisions; never changes physical runtime."""
import numpy as np
from progress import progress
THRESHOLDS=(.90,.93,.95,.968,.98)
TRIGGERS={'none':None,'sensitive':(25,.010),'medium':(50,.010),'conservative':(100,.005)}

class Schedule:
    def __init__(self,positions,goals,threshold,trigger):
        self.initial_distance=np.linalg.norm(np.asarray(goals)-positions,axis=-1)
        self.threshold=float(threshold);self.trigger=trigger;self.history=[]
        self.triggered_at=0 if trigger=='none' else None;self.switched_at=None
    def decide(self,t,positions,goals):
        assert t==len(self.history)
        p=progress(positions,goals,self.initial_distance);self.history.append(p)
        if self.triggered_at is None:
            w,eps=TRIGGERS[self.trigger]
            if t>=w and p-self.history[t-w]<eps:self.triggered_at=int(t)
        if self.triggered_at is None:return 'safety'
        if self.switched_at is None and (p>=self.threshold or t>=600):self.switched_at=int(t)
        return 'early' if self.switched_at is None else 'late'

def offline_trigger(positions,goals,trigger):
    if trigger=='none':return 0
    d0=np.linalg.norm(np.asarray(goals)-positions[0],axis=-1);p=[progress(x,goals,d0) for x in positions];w,eps=TRIGGERS[trigger]
    return next((t for t in range(w,len(positions)-1) if p[t]-p[t-w]<eps),None)

def encode(a):
    from action_encoding import encode as old
    trigger=a.get('trigger','none')
    threshold=a.get('progress_threshold',.9681620884830543 if a['switch_mode']=='progress' else 0.)
    return [*old(a),float(threshold),*[float(trigger==s) for s in TRIGGERS]]
