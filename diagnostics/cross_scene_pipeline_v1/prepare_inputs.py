"""Prespecified redundancy audit, physical inputs and strictly source-only fold data."""
from common import *
from scenes import features,FEATURE_NAMES
from models import select
import numpy as np

OLD=ROOT/'diagnostics/cross_scene_correction_v1'
EXPECTED=['lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_swap8_obstacles','xsc_swap_n4',
    'xsc_swap_n12','xsc_swap_n8_dense','xsc_gap_n4','xsc_gap_n6','xsc_double_n4','xsc_offset_n4']

def rank(v):
    order=np.argsort(v,kind='stable');r=np.empty(len(v));r[order]=np.arange(len(v))
    for val in np.unique(v):r[v==val]=r[v==val].mean()
    return r

def main():
    assert read(HERE/'REPLAY_CHECK.json')['passed']
    scenes=[s for s in read_evidence(OLD/'SCENES.json')if s['family']!='extrapolation'];tables=read_evidence(OLD/'LABEL_TABLES.json')
    selected=read_evidence(OLD/'SELECTED_CONTROLLERS.json');names=[s['name']for s in scenes]
    q=np.array([np.asarray(tables[n]['y'],float).mean((0,1))for n in names]);f=np.array([s['features']for s in scenes])
    rho=np.corrcoef(np.array([rank(v)for v in q]));reg=q.max(1)[:,None]-q;J=np.array([[np.maximum(a,b).min()for b in reg]for a in reg]);keep=set(range(18));dropped=[]
    while True:
        pairs=[(rho[i,j],i,j)for i in keep for j in keep if i<j and rho[i,j]>=.9 and J[i,j]<=.02]
        if not pairs:break
        _,i,j=max(pairs);mi=np.mean([rho[i,k]for k in keep if k!=i]);mj=np.mean([rho[j,k]for k in keep if k!=j]);d=i if mi>=mj else j
        keep.remove(d);dropped.append(dict(dropped=names[d],paired=names[j if d==i else i],rho=float(rho[i,j]),shared_regret=float(J[i,j])))
    kept=[names[i]for i in sorted(keep)]
    write(HERE/'SCENE_SET.json',dict(kept=kept,dropped=dropped,names=names,rho=rho.tolist(),shared_regret=J.tolist(),
        labels_used_to_prespecify_scene_deduplication=True,new_rollouts=0))
    if kept!=EXPECTED:
        write(HERE/'STOP.receipt.json',dict(reason='SCENE_SET_MISMATCH',actual=kept,expected=EXPECTED));raise RuntimeError('SCENE_SET_MISMATCH')
    etas=np.array([a['eta_a']for a in actions()],float);assert np.all(etas>=[0,-1.5,0])and np.all(etas<=[2,1.5,1.5])
    inputs={};physical={}
    for scene in scenes:
        n=scene['name'];tab=tables[n];by={s['index']:s for s in scene['states']};states=[by[i]for i in tab['state_indices']]
        h=np.array([features(dict(scene,states=[s]))[-6:]for s in states]);assert np.allclose(h.mean(0),scene['features'][12:],rtol=0,atol=1e-12)
        controller_types=[]
        for ctl in tab['controllers']:
            row=next(c for c in selected[n].values()if c is not None and c['controller']==ctl);controller_types.append(row['type_code'])
        assert len(controller_types)==len(set(controller_types))
        inputs[n]=dict(h=h.tolist(),c=scene['features'][:12],state_indices=tab['state_indices'],controllers=tab['controllers'],
            controller_types=controller_types,controller_one_hot=[list(np.eye(8)[TYPES.index(t)])for t in controller_types],
            training_state_positions=[j for j,s in enumerate(states)if s['index']%4!=3],
            validation_state_positions=[j for j,s in enumerate(states)if s['index']%4==3],
            state_identity_hashes=[uid('physical',{k:s[k]for k in ('positions','velocities','goals')})for s in states])
        physical[n]=states
    assert sum(len(inputs[n]['controllers'])*len(inputs[n]['state_indices'])for n in kept)==796
    write(HERE/'INPUTS.json',dict(scene_inputs=inputs,state_fields=FEATURE_NAMES[12:],environment_fields=FEATURE_NAMES[:12],
        controller_order=TYPES,eta_keys=[a['key']for a in actions()],etas=etas.tolist(),low=[0,-1.5,0],high=[2,1.5,1.5],
        source_sha256=sha(OLD/'SCENES.json'),labels_not_in_input_features=True))
    write(HERE/'PHYSICAL_STATES.json',physical)
    folds=[]
    for n in kept:folds.append(dict(name='loso__'+n,kind='loso',sources=[v for v in kept if v!=n],targets=[n],generator=True,modes=['full','no_controller','scene_only']))
    bott=[n for n in kept if n.startswith(('xsc_gap','xsc_double','xsc_offset'))];lit=[n for n in kept if n.startswith('lit_')]
    folds.extend([dict(name='lofo__bottleneck',kind='lofo',sources=[n for n in kept if n not in bott],targets=bott,generator=True,modes=['full','no_controller','scene_only']),
        dict(name='lofo__literature',kind='lofo',sources=[n for n in kept if n not in lit],targets=lit,generator=True,modes=['full','no_controller','scene_only'])])
    for n in names:folds.append(dict(name='full18__'+n,kind='full18',sources=[v for v in names if v!=n],targets=[n],generator=False,modes=['full']))
    assert len(folds)==31 and len(bott)==4 and len(lit)==4
    for fold in folds:
        source=fold['sources'];assert not(set(source)&set(fold['targets']))
        train_h=np.vstack([np.array(inputs[n]['h'])[inputs[n]['training_state_positions']]for n in source])
        train_c=np.vstack([np.repeat(np.array(inputs[n]['c'])[None],len(inputs[n]['training_state_positions']),axis=0)for n in source])
        hm,hs=train_h.mean(0),train_h.std(0);cm,cs=train_c.mean(0),train_c.std(0);hs=np.where(hs<1e-6,1,hs);cs=np.where(cs<1e-6,1,cs)
        norm=dict(h_mean=hm.tolist(),h_scale=hs.tolist(),c_mean=cm.tolist(),c_scale=cs.tolist(),
            fit_scenes=source,fit_state_indices={n:[inputs[n]['state_indices'][i]for i in inputs[n]['training_state_positions']]for n in source})
        arrays={}
        for n in source:
            row=inputs[n];arrays[n+'__h']=np.array(row['h']);arrays[n+'__c']=np.repeat(np.array(row['c'])[None],len(row['h']),axis=0)
            arrays[n+'__onehot']=np.array(row['controller_one_hot']);arrays[n+'__y']=np.array(tables[n]['y'],float)
            arrays[n+'__state_indices']=np.array(row['state_indices']);arrays[n+'__train']=np.array(row['training_state_positions']);arrays[n+'__val']=np.array(row['validation_state_positions'])
        folder=HERE/'folds'/fold['name'];folder.mkdir(parents=True,exist_ok=True);path=folder/'source_labels.npz';np.savez_compressed(path,**arrays)
        fold['normalization']=norm;fold['training_data_sha256']=sha(path);fold['source_labels_path']=str(path)
        train=[names.index(n)for n in source];baseline={}
        for n in fold['targets']:
            sel=select(f[train],q[train],f[names.index(n)],etas)
            baseline[n]=dict(select=sel,global_fixed=actions()[sel['global_fixed']]['key'],knn=actions()[sel['knn']]['key'],
                eta13='eta13',safety='safety')
        fold['baselines']=baseline;write(folder/'definition.json',fold)
    write(HERE/'FOLDS.json',folds)
    write(HERE/'INPUTS.receipt.json',dict(utc=utc(),files={str(HERE/n):sha(HERE/n)for n in ('SCENE_SET.json','INPUTS.json','PHYSICAL_STATES.json','FOLDS.json')},
        unique_scene_controller_state_cells=796,source_only_label_files=True,heldout_scene_labels_in_training_files=False,
        training_runs=dict(generator=39,critic=171,total=210),new_rollouts=0))
    print('INPUTS_READY: 11 scenes, 796 cells, 31 folds, 210 seed fits; zero new rollouts.',flush=True)

if __name__=='__main__':main()
