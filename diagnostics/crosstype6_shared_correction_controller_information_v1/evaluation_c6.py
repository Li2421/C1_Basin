"""Freeze models/choices before any fresh closed-loop; evaluate all122 eligible C."""
from common import *
from menu_c6_v2 import METHODS,menu,choices
from schedule_c6 import encode
from actions_c6 import case
from registry import register
from reuse_c6 import infer

def freeze():
    assert read(HERE/'FROZEN_INFERENCE_TESTS.json')['passed'];assert read(HERE/'C5_FROZEN_REFERENCE_TESTS.json')['passed'];selected=read(HERE/'SELECTED_MODELS.json');write(HERE/'FROZEN_PIPELINE.json',dict(utc=utc(),evidence='DEVELOPMENT',outer=selected['outer'],all_types=selected['all_types'],fresh_closed_loop_before_freeze=False,eta_source_decision=read(HERE/'ETA_SOURCE_DECISION.json'),reference_C5_pipeline_sha256=sha(D/'crosstype5_repair_gate_late_eta_v1/FROZEN_PIPELINE.json')))
    seal('FROZEN_PIPELINE.receipt.json',[HERE/n for n in ('FROZEN_PIPELINE.json','SELECTED_MODELS.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','C5_FROZEN_REFERENCE_TESTS.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','TRAINING_HUMAN_ETA_EXCLUSION.receipt.json','ETA_SOURCE_DECISION.json','CANDIDATE_TIE_RULE_CORRECTION.receipt.json','menu_c6_v2.py','evaluation_c6.py','frozen_c5_reference.py')],before_first_fresh_closed_loop=True)

def choose():
    import jax
    from inference import predict
    from frozen_c5_reference import selected as c5_selected
    from actions_c6 import action as human_action
    assert jax.default_backend()=='gpu';base=read(HERE/'BASE_PROTOCOL.json');nn=base['controller_order'];frozen=read(HERE/'FROZEN_PIPELINE.json')
    with np.load(HERE/'CONTROLLER_FEATURES_V2.npz',allow_pickle=False) as z:newfeatures=z['features']
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:oldfeatures=z['features']
    with np.load(HERE/'NEW_EVAL_INPUTS.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    menus=[menu(p) for p in inputs['proposals']];vectors=np.asarray([[encode(a) for a in rows] for rows in menus],np.float32);output=[];predictions=[]
    with jax.default_matmul_precision('highest'):
        for held in sorted(set(base['types'].values())):
            ids=[i for i,n in enumerate(nn) if base['types'][n]==held];model=frozen['outer'][held];assert held not in model['full']['source_types'];data=dict(names=np.asarray(nn)[ids],types=np.asarray([held]*len(ids)),h=inputs['h'],c=inputs['c'],features=newfeatures[ids],action=np.broadcast_to(vectors[None],(len(ids),*vectors.shape)))
            pn=predict(model['no_controller'],data);pf=predict(model['full'],data);reference=c5_selected(held,dict(data,features=oldfeatures[ids]),inputs['proposals'],inputs['state_uids'])
            for ci,ni in enumerate(ids):
                for si in range(24):
                    slots=choices(menus[si],pn[:,ci,si],pf[:,ci,si],model['global_trigger']['setting']);methods={m:menus[si][ai] for m,ai in slots.items()};methods[METHODS[5]]=reference[ci][si];methods[METHODS[6]]=human_action(base['eta'][13],base['eta'][10],.968,'none',name='human_progress_exact0968')
                    assert set(methods)==set(METHODS)
                    output.append(dict(controller=nn[ni],type=held,state_uid=str(inputs['state_uids'][si]),state_index=si,methods=methods,selected_ids=slots,global_trigger=model['global_trigger']['setting'],whole_type_excluded_from_selector_training=True))
            path=HERE/'new_eval_predictions'/f'{held}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,names=np.asarray(nn)[ids],no_controller=pn,with_offline_features=pf);predictions.append(path)
            print('Fresh choices',held,len(ids),'controllers',flush=True)
    assert len(output)==122*24;write(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json',dict(utc=utc(),states=output,method_names=METHODS,outcomes_read=False,evidence='DEVELOPMENT'));write(HERE/'NEW_EVAL_CANDIDATES.json',menus)
    seal('NEW_EVAL_CHOICES.receipt.json',[HERE/n for n in ('NEW_EVAL_CHOSEN_ACTIONS.json','NEW_EVAL_CANDIDATES.json','NEW_EVAL_INPUTS.receipt.json','FROZEN_PIPELINE.receipt.json','FULL_MENU_DEVELOPMENT_VALIDATION.receipt.json','evaluation_c6.py','menu_c6_v2.py','frozen_c5_reference.py')]+predictions,before_first_fresh_closed_loop=True,outcomes_read=False)

def design():
    from actions_v3 import case as old_case
    assert (HERE/'NEW_EVAL_CHOICES.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'NEW_EVAL_STATES.json');rows={};profiles={};links=[]
    for r in read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states']:
        for q in (0,1):
            out={}
            for m,a in r['methods'].items():
                maker=old_case if a.get('c6_schedule') is False else case;c,cp=maker(base,r['controller'],states[r['state_index']],a,q,r['state_index']);rid=c['rollout_uid'];rows.setdefault(rid,c);profiles[cp['controller_uid']]=cp;out[m]=rid
            links.append(dict(controller=r['controller'],type=r['type'],state_index=r['state_index'],seed=q,methods=out))
    assert len(rows)<=46848;register('evaluation',dict(base,states=states,phase='evaluation'),list(rows.values()),profiles);write(HERE/'evaluation_ACTION_LINKS.json',links);missing=read(HERE/'evaluation_MISSING_CASES.json');aliases={}
    with rollout_db.connect(True) as db:
        for c in missing:
            if not c.get('c6_schedule'):continue
            a=infer(c,base,db)
            if a:aliases[c['rollout_uid']]=a
    execution=[c for c in missing if c['rollout_uid'] not in aliases];write(HERE/'evaluation_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'evaluation_EXECUTION_CASES.json',execution);write(HERE/'evaluation_BUDGET.json',dict(utc=utc(),maximum_requests=46848,unique_physical_requests=len(rows),exact_cached=len(rows)-len(missing),local_equivalence=len(aliases),actual_execution_cap=len(execution),controllers=122,types=19))
    seal('evaluation_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('evaluation_protocol.json','evaluation_CASES.json','evaluation_PREFLIGHT.receipt.json','evaluation_ACTION_LINKS.json','evaluation_LOCAL_EQUIVALENCE.json','evaluation_EXECUTION_CASES.json','evaluation_BUDGET.json','NEW_EVAL_CHOICES.receipt.json','FROZEN_PIPELINE.receipt.json','evaluation_c6.py','worker_c6.py','worker_evaluation_c6.py','actions_c6.py','execution_c6.py','schedule_c6.py','reuse_c6.py','execution_v3.py','requery_execution_v3.py')],before_rollout=True)

def assemble():
    assert (HERE/'evaluation_COMPLETE.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');nn=base['controller_order'];shape=(122,24,8,2);label=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown','triggered')};steps=np.full(shape,-1,np.int32);aliases=read(HERE/'evaluation_ALL_LOCAL_EQUIVALENCE.json');cases={c['rollout_uid']:c for c in read(HERE/'evaluation_CASES.json')}
    with rollout_db.connect(True) as db:
        for link in read(HERE/'evaluation_ACTION_LINKS.json'):
            for m,rid in link['methods'].items():
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();r=dict(row) if row else aliases[rid]['outcome'];ix=(nn.index(link['controller']),link['state_index'],METHODS.index(m),link['seed']);assert label[ix]==-2;label[ix]=-1 if r['numerical_failure'] else r['success'];uids[ix]=rid
                for k in ('collision','timeout','unknown'):events[k][ix]=r['numerical_failure'] if k=='unknown' else r[k]
                raw=HERE/'evaluation/raw'/f'{rid}.json';rr=read(raw) if raw.exists() else None
                if rr is not None and rr.get('trigger_step') is not None and cases[rid].get('trigger','none')!='none':steps[ix]=rr['trigger_step'];events['triggered'][ix]=1
    assert not np.any(label==-2);np.savez_compressed(HERE/'NEW_EVAL_BUNDLE.npz',names=nn,types=[base['types'][n] for n in nn],methods=METHODS,labels=label,uids=uids,trigger_steps=steps,**events)
    seal('NEW_EVAL_BUNDLE.receipt.json',[HERE/n for n in ('NEW_EVAL_BUNDLE.npz','evaluation_COMPLETE.receipt.json','evaluation_c6.py','NEW_EVAL_CHOICES.receipt.json')])

if __name__=='__main__':globals()[sys.argv[1]]()
