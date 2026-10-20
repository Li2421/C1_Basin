"""Post-freeze LOTO choices over the full menu; absent labels stay unscored."""
from common import *
from menu_c6_v2 import METHODS,menu,choices
from schedule_c6 import encode
from actions_c6 import case
from train_critic import load_data
from inference import predict
import jax

def main():
    assert jax.default_backend()=='gpu';frozen=read(HERE/'FROZEN_PIPELINE.json');data=load_data();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');menus=[menu(p) for p in data['proposals']];vectors=np.asarray([[encode(a) for a in rows] for rows in menus],np.float32);methods=(*METHODS[:5],METHODS[7]);records=[];results={};paths=[]
    with jax.default_matmul_precision('highest'):
        for held in sorted(set(data['types'])):
            model=frozen['outer'][str(held)];assert held not in model['full']['source_types'];ids=np.flatnonzero(data['types']==held);foldrows=[];pn=None
            for ci in ids:
                one=dict(names=data['names'][ci:ci+1],types=data['types'][ci:ci+1],features=data['features'][ci:ci+1],h=data['h'],c=data['c'],action=vectors[None]);name=str(data['names'][ci])
                if pn is None:pn=predict(model['no_controller'],one)[:,0]
                weight=model['full']['full_weight'];pf=weight*predict(model['full']['full'],one)[:,0]+(1-weight)*pn
                folder=HERE/'fullmenu_development_predictions';folder.mkdir(exist_ok=True);p=folder/f'{name}.npz';np.savez_compressed(p,no_controller=pn,with_offline_features=pf);paths.append(p)
                for si,s in enumerate(states):
                    selected=choices(menus[si],pn[:,si],pf[:,si],model['global_trigger']['setting']);links={};labels={};support={};known_uids={str(uid):ai for ai,uid in enumerate(data['uids'][ci,si,:,0])}
                    for method in methods:
                        a=menus[si][selected[method]];c,_=case(base,name,s,a,0,si);rid=c['rollout_uid'];slot=known_uids.get(rid);value=None if slot is None else int(data['labels'][ci,si,slot,0]);links[method]=rid;labels[method]=value if value is not None and value>=0 else None;support[method]='absent_label' if value is None or value==-2 else 'numerical_unknown' if value==-1 else 'scored'
                    row=dict(controller=name,type=str(held),state_index=si,state_uid=s['state_uid'],selected_uids=links,known_success=labels,support=support,whole_type_excluded=True);records.append(row);foldrows.append(row)
            summaries={}
            for method in methods:
                per=[];scored=0;absent=0;numerical=0
                for name in data['names'][ids]:
                    rr=[r for r in foldrows if r['controller']==name];valid=[r['known_success'][method] for r in rr if r['known_success'][method] is not None]
                    if valid:per.append(float(np.mean(valid)))
                    scored+=len(valid);absent+=sum(r['support'][method]=='absent_label' for r in rr);numerical+=sum(r['support'][method]=='numerical_unknown' for r in rr)
                summaries[method]=dict(success_on_scored_states_controller_equal=float(np.mean(per)) if per else None,scored_states=scored,scored_controllers=len(per),unscored_fraction=1-scored/len(foldrows),absent_label_states=absent,numerical_unknown_states=numerical,not_a_full_population_success_estimate=True)
            results[str(held)]=dict(source_types=model['full']['source_types'],controllers=len(ids),states=len(foldrows),methods=summaries,checkpoint_and_global_trigger_previously_frozen=True)
            print('Full-menu LOTO coverage',held,len(foldrows),'states',flush=True)
    write(HERE/'FULL_MENU_DEVELOPMENT_CHOICES.json',records);write(HERE/'FULL_MENU_DEVELOPMENT_VALIDATION.json',dict(utc=utc(),folds=results,candidates_per_state=1313,model_checkpoint_blend_global_trigger_frozen_before_validation=True,missing_labels_not_failure_and_not_imputed=True,source_calibration='Global trigger was selected using the preregistered labeled53-slot source-only calibration; this separate full-menu validation is not fed back into any selection',ranking='DEVELOPMENT_OUTER_FOLDS.json retains ranking on labeled actions including theta/trigger groups'))
    seal('FULL_MENU_DEVELOPMENT_VALIDATION.receipt.json',[HERE/n for n in ('FULL_MENU_DEVELOPMENT_CHOICES.json','FULL_MENU_DEVELOPMENT_VALIDATION.json','fullmenu_development_c6.py','menu_c6_v2.py','TRAINING_BUNDLE.receipt.json','FROZEN_PIPELINE.receipt.json')]+paths,post_model_freeze=True,no_extra_rollouts=True,absent_labels_unscored=True)

if __name__=='__main__':main()
