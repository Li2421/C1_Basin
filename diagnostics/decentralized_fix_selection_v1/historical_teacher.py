"""Retain actual historical teacher features without relabelling their probe protocol."""
from support import *
G=guard()
def freeze():
 assert not list((HERE/'folds').glob('*/CHOICES_FROZEN.receipt.json'))
 path=D/'teacher_student_macflow_v1/DEVELOPMENT_TEACHER_CHOICES.json';r=read(path);source=Path(r['descriptor_source']);assert sha(source)==r['source_hashes'][str(source)];z=np.load(source);names=[q['name'] for q in read(HERE/'TEACHERS.json')];ids=[list(z['controller_names']).index(n) for n in names];np.savez_compressed(HERE/'RETAINED_TEACHER_HISTORICAL_R5.npz',features=z['features'][ids],controller_names=[f'TD{i:02}' for i in range(14)],original_names=names)
 rows=[dict(name=f'TD{i:02}',original_name=n,original_choice=r['controllers'][n]['A3'],descriptor_protocol=r['descriptor_pipeline']) for i,n in enumerate(names)];write(HERE/'RETAINED_TEACHER_HISTORICAL_CRITIC.json',dict(rows=rows,source=str(path),source_sha256=sha(path),R5_source=str(source),R5_sha256=sha(source),protocol_compatible_with_new_Dlocal_R5=False,usage='Descriptive comparator using the original frozen native/joint-flow R5 protocol; never substituted for the missing exact distributed-teacher R5 and never used by new selectors.'))
 seal('TEACHER_HISTORICAL_FEATURES_FROZEN.receipt.json',[HERE/'RETAINED_TEACHER_HISTORICAL_R5.npz',HERE/'RETAINED_TEACHER_HISTORICAL_CRITIC.json'],before_any_selector=True,new_probes=0,new_predictions=0,guard=guard());source_inventory('imports_historical_teacher_freeze.json');print('Retained 14 existing teacher R5 rows and original frozen choices; original protocol explicitly retained.',flush=True)
def compare():
 z=np.load(HERE/'LABELS.npz');names=list(z['controller_names']);fam=z['families'];y=z['labels'];old=read(HERE/'FROZEN_OLD_CRITIC_REUSE.json');teacher=read(HERE/'RETAINED_TEACHER_HISTORICAL_CRITIC.json')['rows'];choices={r['name']:r['original_choice'] for r in old+teacher};history=read(D/'teacher_student_line_closeout_v1/MANIFEST.json');ys=np.full((46,32,2),-9,np.int8);rows=[]
 for i,n in enumerate(names):
  arm=choices[n]
  if arm in ARMS:ys[i]=y[i,ARMS.index(arm)]
  else:
   group='V0' if n.startswith('ST') else 'TD';idx=int(n[2:]);rr=[r for r in history if r['group']==group and r['controller_index']==idx and r['arm']==arm];assert len(rr)==64
   for r in rr:ys[i,r['state_index'],r['seed']]=-1 if r['unknown'] else r['success']
  rows.append(dict(name=n,arm=arm,descriptor_protocol='original joint-flow historical R5; not D-local R5' if n.startswith('TD') else 'existing distributed R5',success_lower=float((ys[i]==1).mean()),success_upper=float((ys[i]!=0).mean()),unknown=int((ys[i]==-1).sum())))
 assert not (ys==-9).any();rng=np.random.default_rng(2026101708);lo=(ys==1).mean(-1);hi=(ys!=0).mean(-1);draw=[];upper=[]
 for b in range(5000):
  ids=np.concatenate([rng.choice(np.flatnonzero(fam==f),sum(fam==f),replace=True) for f in FAMILIES]);ss=rng.integers(0,32,(46,32));draw.append(lo[ids[:,None],ss].mean());upper.append(hi[ids[:,None],ss].mean())
 write(HERE/'HISTORICAL_PROTOCOL_OLD_CRITIC_COMPARATOR.json',dict(evidence_level='DEVELOPMENT',coverage=46,rows=rows,success_lower=float(lo.mean()),success_upper=float(hi.mean()),ci=np.quantile(draw,[.025,.975]),unknown_ci_envelope=[float(np.quantile(draw,.025)),float(np.quantile(upper,.975))],families={f:dict(success_lower=float(lo[fam==f].mean()),success_upper=float(hi[fam==f].mean())) for f in FAMILIES},bootstrap=dict(replicates=5000,seed=2026101708,paired_state_clusters=True),protocol_warning='14 teachers retain their original historical joint-flow R5. This is an as-designed historical frozen-critic comparator, not exact D-local R5 transfer. New selectors never consume these descriptors.',guard=guard()));source_inventory('imports_historical_critic_compare.json')
if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else compare()
