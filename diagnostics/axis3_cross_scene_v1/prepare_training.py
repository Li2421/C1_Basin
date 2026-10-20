from common import *
from features import physical12,action7

def main():
 prereg();assert (HERE/'R5_FROZEN.receipt.json').exists();z=np.load(HERE/'R5_FEATURES.npz');keys=z['controller_keys'].tolist();feat={k:z['features'][i] for i,k in enumerate(keys)};feat['toy_give_way::flow_isotropic']=z['Toy_source_isotropic'];inv=read(HERE/'HISTORICAL_INVENTORY.json');records=[];paths={};membership={};raws={};physical_cache={};db=rollout_db.connect(True)
 for sc in SCENES:
  if sc=='four_way_intersection':
   sources=[(R19/'FIXED_CASES.json',R19/'FIXED_PROTOCOL.json',R19/'stage2/raw'),(R19/'SCHEDULE_CASES.json',R19/'SCHEDULE_PROTOCOL.json',R19/'stage3/raw')];aliases={'S15to10_471':'S15_t1','S15to10_786':'S15_t2','S13to10_786':'S13_t2','F10':'F10','F13':'F13','F15':'F15'};names=inv['scenes'][sc]['controllers']
  else:sources=[(R13/'main_CASES.json',R13/'protocol.json',R13/'main/raw')];aliases={a:a for a in MENU};names=['flow_isotropic'] if sc=='toy_give_way' else inv['scenes'][sc]['controllers']
  membership[sc]=dict(controllers=names,source_rows=0,source_trials=0,source_unknown=0)
  for cp,pp,rawdir in sources:
   cases=read(cp);protocol=read(pp);states={s['state_uid']:s for s in protocol['states']};paths[str(cp)]=sha(cp);paths[str(pp)]=sha(pp)
   for c in cases:
    if c.get('scene')!=sc or c['base_controller_name'] not in names or c['arm'] not in aliases:continue
    arm=aliases[c['arm']];key=sc+'::'+c['base_controller_name'];assert key in feat,'Missing source R5 '+key;rpath=rawdir/f"{c['rollout_uid']}.json";cached=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
    if cached is not None:r=dict(cached);assert not r['conflict_quarantined']
    else:r=read(rpath);assert r['case']==c
    if (sc,c['state_uid']) not in physical_cache:physical_cache[sc,c['state_uid']]=physical12(sc,states[c['state_uid']])
    ix=(sc,key,c['state_uid'],arm)
    if ix not in raws:raws[ix]=dict(state=states[c['state_uid']],success=0,known=0,total=0,source_uids=[],x=np.concatenate([physical_cache[sc,c['state_uid']],feat[key],action7(arm)]))
    raws[ix]['success']+=r['success'];raws[ix]['known']+=1-r['numerical_failure'];raws[ix]['total']+=1;raws[ix]['source_uids'].append(c['rollout_uid']);membership[sc]['source_trials']+=1;membership[sc]['source_unknown']+=r['numerical_failure']
 for (sc,key,sid,arm),v in raws.items():records.append(dict(scene=sc,controller_key=key,state_uid=sid,arm=arm,total=v['total'],source_uids=v['source_uids']));membership[sc]['source_rows']+=1
 X=np.asarray([v['x'] for v in raws.values()],np.float32);y=np.asarray([v['success'] for v in raws.values()],np.float32);n=np.asarray([v['known'] for v in raws.values()],np.float32);q=np.asarray([v['total'] for v in raws.values()],np.float32);scenes=np.asarray([r['scene'] for r in records]);assert X.shape[1]==24 and set(scenes.tolist())==set(SCENES);np.savez_compressed(HERE/'SOURCE_TRAINING_DATA.npz',X=X,y=y,n=n,q=q,scenes=scenes,controller_keys=[r['controller_key'] for r in records],state_uids=[r['state_uid'] for r in records],arms=[r['arm'] for r in records]);write(HERE/'SOURCE_LABEL_PROVENANCE.json',dict(utc=utc(),source_files=paths,membership=membership,rows=records,grade='Historical source labels retain original grade; no new target labels in fit',Toy_precision='HistoricalFP64 only; not relabeled FP32'));seal('TRAINING_INPUTS_FROZEN.receipt.json',[HERE/'SOURCE_TRAINING_DATA.npz',HERE/'SOURCE_LABEL_PROVENANCE.json',HERE/'R5_FROZEN.receipt.json',HERE/'features.py',HERE/'prepare_training.py'],no_new_evaluation_labels=True);print('Source-only training',X.shape,membership,flush=True)
if __name__=='__main__':main()
