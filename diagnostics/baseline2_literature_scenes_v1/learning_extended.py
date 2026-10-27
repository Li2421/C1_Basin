"""Supplementary ten-type branch; reuse all original eight-type evidence."""
from experiment import *
from registry import register
from learning_design import selected_states,encode,make_case as original_case
from pilot_design import manifest,make_case as pilot_case
from flow_scene import physical12
import numpy as np
import collections

def path_for(rid):
    matches=list(HERE.glob(f'*/raw/{rid}.json'))
    if not matches:raise FileNotFoundError(rid)
    return sorted(matches)[0]

def newcase(s,ctl,a,ai,seed):
    if ctl['type_code'] in ('deterministic','mixture'):
        c=pilot_case(s,ctl,dict(a,name='safety' if ai==0 else f'learning_action_{ai:02d}'),seed);c['menu_index']=ai;return c
    return original_case(s,ctl,a,ai,seed)

def plan(scene):
    prereg();q=read(HERE/f'pilot_capability_{scene}_QUALIFICATION.json')
    if not any(r['qualified'] for r in q):write(HERE/f'PILOT_LEARNING_GAP_{scene}.json',dict(reason='No BC checkpoint passed the unchanged95% single-robot gate',required_by_protocol=False));return
    informative=read(HERE/f'BASELINE_USABILITY_{scene}.json')['informative'] or read(HERE/f'PILOT_BASELINE_USABILITY_{scene}.json')['informative']
    if not informative:write(HERE/f'PILOT_LEARNING_GAP_{scene}.json',dict(reason='Scene only requires fixed correction under the preregistered98% usability rule',required_by_protocol=False));return
    states=selected_states(scene,'train');groups=collections.defaultdict(list)
    for c in read(HERE/'CONTROLLER_MANIFEST.json'):groups[c['type_code']].append(c)
    ctls=[c for t in sorted(groups) for c in groups[t][:2]]+manifest(scene);menu=read(HERE/'LEARNING_MENU.json');cases=[newcase(s,ctl,a,ai,0) for ctl in ctls for s in states for ai,a in enumerate(menu)]
    original_manifest=HERE/f'learning_labels_{scene}_CASES.json'
    if original_manifest.exists():
        old_ids={c['rollout_uid'] for c in read(original_manifest)};same_ids={c['rollout_uid'] for c in cases if c['controller_payload']['nominal']['type_code'] not in ('deterministic','mixture')}
        assert old_ids==same_ids,'A critic/type extension must preserve every original execution request identity'
    design=dict(scene=scene,tag=scene+'_with_bc',training_states=states,evaluation_states=selected_states(scene,'evaluation'),new_types=['deterministic','mixture'],training_controllers=ctls,new_controllers=manifest(scene),original_labels_reused=str(HERE/f'LEARNING_BUNDLE_{scene}.npz'),source_rules='ten-type leave-one-type-out, nested source-only checkpoint selection; all other critic hyperparameters unchanged',logical_requests=len(cases))
    p=HERE/f'PILOT_LEARNING_DESIGN_{scene}.json';write(p,design);seal(f'PILOT_LEARNING_DESIGN_{scene}.receipt.json',[p,HERE/'learning_extended.py',HERE/'learn_critic_extended.py',HERE/'extended_nominal.py',HERE/'flow_runtime_extended.py'],before_new_type_labels=True)
    register('pilot_learning_labels_'+scene,cases)

def assemble(scene):
    tag=scene+'_with_bc';states=selected_states(scene,'train');old=HERE/f'LEARNING_BUNDLE_{scene}.npz'
    ctls=read(HERE/f'PILOT_LEARNING_DESIGN_{scene}.json')['training_controllers'];labels=np.full((len(ctls),32,12),-1,np.int8);ci={c['controller']:i for i,c in enumerate(ctls)};si={(s['N'],s['index']):i for i,s in enumerate(states)};sources=[]
    for c in read(HERE/f'pilot_learning_labels_{scene}_CASES.json'):
        p=path_for(c['rollout_uid']);r=read(p);i=ci[c['controller_payload']['nominal']['controller']];j=si[(c['state']['N'],c['state']['index'])];labels[i,j,c['menu_index']]=-1 if r['numerical_failure'] else r['success'];sources.append(dict(rollout_uid=c['rollout_uid'],path=str(p),sha256=sha(p)))
    data=dict(labels=labels,controllers=np.asarray([c['controller'] for c in ctls]),types=np.asarray([c['type_code'] for c in ctls]),physical12=np.asarray([physical12(s) for s in states]),action=np.asarray([encode(a) for a in read(HERE/'LEARNING_MENU.json')]))
    p=HERE/f'LEARNING_BUNDLE_{tag}.npz';np.savez_compressed(p,**data);write(HERE/f'LEARNING_BUNDLE_SOURCES_{tag}.json',dict(original_bundle_sha256=sha(old) if old.exists() else None,sources=sources))
    paths=[p,HERE/f'LEARNING_BUNDLE_SOURCES_{tag}.json'];receipt=HERE/f'LEARNING_BUNDLE_{scene}.receipt.json'
    if receipt.exists():paths.append(receipt)
    seal(f'LEARNING_BUNDLE_{tag}.receipt.json',paths,before_training=True,original_evidence_not_rerun=True)

def freeze(scene):
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from learn_critic_extended import Critic,apply,predict,SEEDS
    tag=scene+'_with_bc';frozen=read(HERE/f'FROZEN_SCENE_CRITIC_{tag}.json');states=selected_states(scene,'evaluation');ctls=read(HERE/'CONTROLLER_MANIFEST.json')+manifest(scene);menu=read(HERE/'LEARNING_MENU.json')
    X=np.asarray([physical12(s) for s in states]);actions=np.asarray([encode(a) for a in menu]);choices={};cases=[]
    template=jax.tree_util.tree_map(lambda *a:jnp.stack(a),*[Critic().init(jax.random.PRNGKey(seed),jnp.zeros((1,9)),jnp.zeros((1,3)),jnp.zeros((1,19))) for seed in SEEDS])
    for t,fold in frozen['folds'].items():
        p=Path(fold['checkpoint']);assert sha(p)==fold['checkpoint_sha256'];params=serialization.from_bytes(template,p.read_bytes());prob=np.asarray(predict(params,jnp.asarray(apply(X,fold['transform'])),jnp.asarray(actions))).mean(0);choices[t]=np.argmax(prob,axis=-1).tolist()
    p=HERE/f'LEARNED_EVAL_CHOICES_{tag}.json';write(p,dict(utc=utc(),states=states,choices_by_held_out_type=choices,frozen_model_sha256=sha(HERE/f'FROZEN_SCENE_CRITIC_{tag}.json'),outcomes_opened=False))
    seal(f'LEARNED_EVAL_CHOICES_{tag}.receipt.json',[p,HERE/'learning_extended.py'],before_evaluation_preflight=True)
    for ctl in ctls:
        for i,s in enumerate(states):
            ai=choices[ctl['type_code']][i]
            for seed in (0,1):
                c=newcase(s,ctl,menu[ai],ai,seed);c['method']='learned_scene';cases.append(c)
    register('learning_eval_'+tag,cases)

if __name__=='__main__':{'plan':plan,'assemble':assemble,'freeze':freeze}[sys.argv[1]](sys.argv[2])
