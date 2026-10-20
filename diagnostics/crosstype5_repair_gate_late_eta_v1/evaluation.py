"""Frozen folds and gates, sealed fresh choices, exact evaluation preflight."""
from common import *
from candidate_menu import menu,choose,METHODS
from action_encoding import encode
from actions_v3 import action,case
from registry import register
from local_reuse import infer

def freeze():
    assert (HERE/'coverage_COMPLETE.receipt.json').exists()
    models=read(HERE/'SELECTED_MODELS.json');gates=read(HERE/'SELECTED_GATES.json')
    frozen=dict(utc=utc(),evidence='DEVELOPMENT',outer={t:dict(**m,gates=gates['outer'][t]) for t,m in models['outer'].items()},all_types=dict(**models['all_types'],gates=gates['all_types']),fresh_closed_loop_before_freeze=False)
    write(HERE/'FROZEN_PIPELINE.json',frozen)
    seal('FROZEN_PIPELINE.receipt.json',[HERE/n for n in ('FROZEN_PIPELINE.json','SELECTED_MODELS.receipt.json','SELECTED_GATES.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','coverage_COMPLETE.receipt.json','candidate_menu.py','evaluation.py','gate_selection.py','schedule_generator.py','critic_model.py','inference.py')],before_first_fresh_rollout_submission=True,all_development=True)

def names():
    # Historical evaluation identity only; historical outcomes are never mixed in.
    p=D/'crosstype4_no_repair_schedule_diagnosis_v1/NEW_EVAL_CHOSEN_ACTIONS.json'
    result=list(dict.fromkeys(r['controller'] for r in read(p)['states']));assert len(result)==74;return result

def choices():
    import jax
    from schedule_generator import predict as generate
    from inference import predict
    for p,h in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(p)==h,p
    base=read(HERE/'BASE_PROTOCOL.json');nn=names();frozen=read(HERE/'FROZEN_PIPELINE.json')
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:features=np.asarray([z['features'][list(z['names']).index(n)] for n in nn])
    with np.load(HERE/'NEW_EVAL_INPUTS.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    output=[];all_menus=[]
    with jax.default_matmul_precision('highest'):
        for held in sorted({base['types'][n] for n in nn}):
            ids=[i for i,n in enumerate(nn) if base['types'][n]==held];model=frozen['outer'][held];assert held not in model['full']['source_types'] and held not in model['generator']['source_types'] and held not in model['gates']['source_types']
            data=dict(names=np.asarray(nn)[ids],types=np.asarray([held]*len(ids)),features=features[ids],h=inputs['h'],c=inputs['c']);proposals=generate(model['generator'],data,inputs['state_uids']);menus=[[menu(inputs['proposals'][si],proposals[ci,si]) for si in range(24)] for ci in range(len(ids))]
            vectors=np.zeros((len(ids),24,415,14),np.float32)
            for ci in range(len(ids)):
                for si in range(24):vectors[ci,si,:len(menus[ci][si])]=[encode(r['action']) for r in menus[ci][si]]
            probability=predict(model['full'],dict(data,action=vectors));transform=read(model['full']['full']['transform_path']);filled=np.where(np.isfinite(data['features']),data['features'],np.asarray(transform['feature_fill']));g=model['gates']
            for ci,ni in enumerate(ids):
                for si in range(24):
                    rows=menus[ci][si];p=probability[:,ci,si,:len(rows)];selected,details=choose(rows,p,g['critic']['k'],g['critic']['delta'],g['probe']['tau'],g['probe']['rho'],filled[ci]);methods={m:rows[index]['action'] for m,index in selected.items()};methods[METHODS[7]]=action(base['eta'][13],base['eta'][10],'fixed',500);methods[METHODS[8]]=action(base['eta'][13],base['eta'][10],'progress',700)
                    output.append(dict(controller=nn[ni],type=held,state_uid=str(inputs['state_uids'][si]),state_index=si,methods=methods,details=details,selected_metadata={m:rows[i] for m,i in selected.items()},whole_type_excluded_from_selector_training=True))
                    all_menus.append(dict(controller=nn[ni],type=held,state_index=si,state_uid=str(inputs['state_uids'][si]),rows=rows,probability=p))
    assert len(output)==74*24
    write(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json',dict(utc=utc(),states=output,method_names=METHODS,outcomes_read=False,evidence='DEVELOPMENT'))
    write(HERE/'NEW_EVAL_MENUS.json',all_menus)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/n for n in ('NEW_EVAL_CHOSEN_ACTIONS.json','NEW_EVAL_MENUS.json','NEW_EVAL_INPUTS.receipt.json','FROZEN_PIPELINE.receipt.json','evaluation.py','candidate_menu.py')],before_first_fresh_rollout_submission=True,outcomes_read=False)

def design():
    assert (HERE/'NEW_EVAL_CHOICES.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'NEW_EVAL_STATES.json');rows={};profiles={};links=[]
    for r in read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states']:
        for q in (0,1):
            out={}
            for method,a in r['methods'].items():
                c,cp=case(base,r['controller'],states[r['state_index']],a,q,r['state_index']);rid=c['rollout_uid'];rows.setdefault(rid,c);profiles[cp['controller_uid']]=cp;out[method]=rid
            links.append(dict(controller=r['controller'],type=r['type'],state_index=r['state_index'],seed=q,methods=out))
    assert len(rows)<=31968
    register('evaluation',dict(base,states=states,phase='evaluation'),list(rows.values()),profiles);write(HERE/'evaluation_ACTION_LINKS.json',links)
    missing=read(HERE/'evaluation_MISSING_CASES.json');aliases={}
    with rollout_db.connect(True) as db:
        for c in missing:
            a=infer(c,base,db)
            if a:aliases[c['rollout_uid']]=a
    execution=[c for c in missing if c['rollout_uid'] not in aliases];write(HERE/'evaluation_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'evaluation_EXECUTION_CASES.json',execution);write(HERE/'evaluation_BUDGET.json',dict(utc=utc(),maximum_requests=31968,unique_physical_requests=len(rows),exact_cached=len(rows)-len(missing),early_equivalent=len(aliases),actual_execution_cap=len(execution)))
    seal('evaluation_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('evaluation_protocol.json','evaluation_CASES.json','evaluation_PREFLIGHT.receipt.json','evaluation_ACTION_LINKS.json','evaluation_LOCAL_EQUIVALENCE.json','evaluation_EXECUTION_CASES.json','evaluation_BUDGET.json','evaluation.py','worker.py','actions_v3.py','execution_v3.py','requery_execution_v3.py','requery_client.py','requery_helper.py','snapshot_conditioning.py','requery_generator_v2.py','NEW_EVAL_CHOICES.receipt.json','FROZEN_PIPELINE.receipt.json')],before_submission=True)

def assemble():
    assert (HERE/'evaluation_COMPLETE.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');nn=names();shape=(74,24,9,2);labels=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown')};aliases=read(HERE/'evaluation_ALL_LOCAL_EQUIVALENCE.json')
    with rollout_db.connect(True) as db:
        for link in read(HERE/'evaluation_ACTION_LINKS.json'):
            for m,rid in link['methods'].items():
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();r=dict(row) if row else aliases[rid]['outcome'];ix=(nn.index(link['controller']),link['state_index'],METHODS.index(m),link['seed']);assert labels[ix]==-2;labels[ix]=-1 if r['numerical_failure'] else r['success'];uids[ix]=rid
                for k in events:events[k][ix]=r['numerical_failure'] if k=='unknown' else r[k]
    assert not np.any(labels==-2);np.savez_compressed(HERE/'NEW_EVAL_BUNDLE.npz',names=nn,types=[base['types'][n] for n in nn],methods=METHODS,labels=labels,uids=uids,**events)
    seal('NEW_EVAL_BUNDLE.receipt.json',[HERE/n for n in ('NEW_EVAL_BUNDLE.npz','evaluation_COMPLETE.receipt.json','evaluation.py','NEW_EVAL_CHOICES.receipt.json')])
if __name__=='__main__':globals()[sys.argv[1]]()
