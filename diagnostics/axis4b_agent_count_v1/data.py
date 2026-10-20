from common import *
ARMS=['nominal_only','safety_baseline','F10','F13','F15','S13t1','S13t2','S15t1','S15t2']
def load_labels(stage,N):
 cases=[c for c in read(HERE/f'{stage}_CASES.json') if c['N']==N];names=[k for k in ['Flow','P2'] if any(c['kind']==k for c in cases)];shape=(len(names),32,9,2);lo=np.zeros(shape);hi=lo.copy();events={k:lo.copy() for k in ['collision','timeout','unknown']};filled=np.zeros(shape,bool)
 for c in cases:
  q=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");idx=(names.index(c['kind']),c['state_index'],ARMS.index(c['arm']),c['seed']);assert not filled[idx];filled[idx]=True;lo[idx]=q['success'];hi[idx]=q['success']+q['numerical_failure'];events['collision'][idx]=q['collision'];events['timeout'][idx]=q['timeout'];events['unknown'][idx]=q['numerical_failure']
 assert filled.all(),(stage,N,int((~filled).sum()));return dict(N=N,classes=names,lower=lo,upper=hi,events=events)
