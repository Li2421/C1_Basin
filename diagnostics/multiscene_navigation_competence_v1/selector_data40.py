"""Separate per-family label files and label-free prediction inputs."""
from common import *
from selector_features40 import StaticMap,OccupancyGrid,physical12,action9
from corrective_design40 import identities,require_ready
from collect39 import checkpoint
from shared_rollout_db.src.rollout_db import connect,canonical
from design_generated import ARMS
MENU=ARMS[2:]

def generated_inputs(family):
    require_ready(family);gate=read(HERE/f'MULTIAGENT_CAPABILITY_{family}.json');assert gate['family_passed']
    probe=read(HERE/f'probe40_{family}_FEATURES.json')['features']
    states=sorted([s for s in read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog'][family] if s['role']=='evaluation'],key=lambda s:s['index']);assert len(states)==32
    geometry_cache={};state_features=[];state_ids=[]
    _,first_kind,first_seed=gate['eligible'][0].split('_');first_cp=checkpoint(family,first_kind,int(first_seed),3)
    for s in states:
        cfg=s['config'];key=canonical(dict(rectangles=s['rectangles'],config={k:v for k,v in cfg.items() if k not in ['seed','max_steps']}))
        if key not in geometry_cache:
            m=StaticMap(cfg,s['rectangles']);geometry_cache[key]=(m,OccupancyGrid(m))
        m,grid=geometry_cache[key]
        state_features.append(np.r_[physical12(s['positions'],s['velocities'],s['goals'],m),grid.features(s['positions'],s['goals'],s['H'],cfg['dt'])]);state_ids.append(identities(s,'F10',0,first_cp)[0]['state_uid'])
    X=[]
    for name in gate['eligible']:
        behavior=np.asarray([np.nan if v is None else v for v in probe[name]],np.float32)
        X.append([[np.r_[sf,behavior,action9(arm)] for arm in MENU] for sf in state_features])
    output=HERE/'selector_inputs40'/f'{family}.npz';output.parent.mkdir(exist_ok=True)
    np.savez_compressed(output,X=np.asarray(X,np.float32),controller_names=gate['eligible'],state_uids=state_ids,arms=MENU,state_indices=[s['index'] for s in states])
    seal(f'{family}_SELECTOR_INPUTS40_FROZEN.receipt.json',[output,HERE/f'probe40_{family}_FEATURES_FROZEN.receipt.json',HERE/'selector_features40.py',HERE/'selector_data40.py'],task_outcomes_read=0,before_target_evaluation_labels_opened=True)
    return output

def generated_labels(family):
    # Packaging is not analysis. Each target's fitting worker can open only other families.
    require_ready(family);stage=f'evaluation40_{family}';assert (HERE/f'{stage}_POSTFLIGHT.json').exists()
    z=np.load(HERE/'selector_inputs40'/f'{family}.npz');names=z['controller_names'].tolist();sids=z['state_uids'].tolist()
    cases=read(HERE/f'{stage}_CASES.json');Y=np.zeros((len(names),32,9,2),np.float32);unknown=np.zeros_like(Y);collision=np.zeros_like(Y);timeout=np.zeros_like(Y);seen=set();provenance=[]
    with connect(True) as db:
        for c in cases:
            ix=(names.index(c['controller_name']),sids.index(c['state_uid']),ARMS.index(c['arm']),c['seed']);assert ix not in seen;seen.add(ix)
            r=dict(db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone());assert not r['conflict_quarantined']
            Y[ix]=r['success'];unknown[ix]=r['numerical_failure'];collision[ix]=r['collision'];timeout[ix]=r['timeout'];provenance.append(dict(rollout_uid=c['rollout_uid'],raw_record_hash=r['raw_record_hash'],controller_uid=c['controller_uid'],state_uid=c['state_uid'],arm=c['arm'],seed=c['seed'],source_scene=family))
    assert len(seen)==Y.size
    folder=HERE/'selector_labels40';folder.mkdir(exist_ok=True);out=folder/f'{family}.npz'
    np.savez_compressed(out,X=z['X'].reshape(-1,33),y=Y[:,:,2:].sum(-1).reshape(-1),n=(2-unknown[:,:,2:].sum(-1)).reshape(-1),q=np.full(z['X'].shape[:-1],2).reshape(-1),success=Y,unknown=unknown,collision=collision,timeout=timeout,controller_names=names,state_uids=sids,arms=ARMS,scene=family)
    p=folder/f'{family}_PROVENANCE.json';write(p,dict(source_scene=family,rows=provenance,grade='CONFIRMATORY conditional on frozen per-target source exclusion and pre-analysis choices',unknown_retries=0))
    seal(f'{family}_SELECTOR_LABELS40_PACKAGED.receipt.json',[out,p,HERE/f'{stage}_POSTFLIGHT.json'],target_file_not_opened_by_its_fit=True,G_never_source=True)

def historical_labels():
    assert (HERE/'HISTORICAL_EVIDENCE40_FROZEN.receipt.json').exists()
    rows=read(HERE/'HISTORICAL_COMPATIBLE_LABELS40.json');behavior=read(HERE/'HISTORICAL_PROBE_FEATURES40.json')['features'];grids={};physical={};aggregates={}
    for row in rows:
        scene=row['scene'];assert scene=='ring_exchange' # Exact-compatible cohort, no invented DB/Four labels.
        metadata=json.loads(row['goals_geometry_json']);cfg=metadata.get('config',metadata);geometry_key=canonical({k:v for k,v in cfg.items() if k not in ['max_steps','seed']})
        if geometry_key not in grids:
            m=StaticMap(cfg,scene=scene);grids[geometry_key]=(m,OccupancyGrid(m))
        m,grid=grids[geometry_key];sid=row['state_uid'];key=row['nominal_key'];H=int(row['profile']['horizon']);cache_key=(sid,H)
        if cache_key not in physical:
            state=row['physical'];physical[cache_key]=np.r_[physical12(state['positions'],state['velocities'],state['goals'],m),grid.features(state['positions'],state['goals'],H,cfg['dt'])]
        ix=(key,sid,row['arm']);value=aggregates.setdefault(ix,dict(x=np.r_[physical[cache_key],np.asarray([np.nan if v is None else v for v in behavior[key]],np.float32),action9(row['arm'])],y=0,n=0,q=0,source_uids=[]))
        value['y']+=row['success'];value['n']+=1-row['numerical_failure'];value['q']+=1;value['source_uids'].append(row['rollout_uid'])
    folder=HERE/'selector_labels40';folder.mkdir(exist_ok=True);out=folder/'ring_exchange.npz';keys=list(aggregates);values=list(aggregates.values())
    np.savez_compressed(out,X=np.asarray([v['x'] for v in values],np.float32),y=[v['y'] for v in values],n=[v['n'] for v in values],q=[v['q'] for v in values],controller_names=[k[0] for k in keys],state_uids=[k[1] for k in keys],arms=[k[2] for k in keys],scene='ring_exchange')
    p=folder/'ring_exchange_PROVENANCE.json';write(p,dict(scene='ring_exchange',grade='Retains original historical evidence grade; not new confirmatory labels',rows=[dict(nominal_key=k[0],state_uid=k[1],arm=k[2],source_uids=v['source_uids'],known=v['n'],total=v['q']) for k,v in zip(keys,values)],missing_scenes=['double_bottleneck','four_way_intersection'],new_rollouts=0))
    seal('ring_exchange_SELECTOR_LABELS40_PACKAGED.receipt.json',[out,p,HERE/'HISTORICAL_EVIDENCE40_FROZEN.receipt.json'],before_generated_evaluation_labels_opened=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
