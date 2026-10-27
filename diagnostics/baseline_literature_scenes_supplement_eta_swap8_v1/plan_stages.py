import bootstrap
from experiment import *
from case_factory import *
from evidence_store import load
from collections import defaultdict
from itertools import permutations

def qualify():
    cases=read(HERE/'capability_CASES.json');groups=defaultdict(list)
    for c in cases:groups[(c['state']['scene'],c['controller_payload']['nominal']['controller'])].append(load('capability',c))
    rows=[]
    for (scene,ctl),rr in sorted(groups.items()):
        assert len(rr)==512;success=sum(r['success'] for r in rr);unknown=sum(r['numerical_failure'] for r in rr)
        rows.append(dict(scene=scene,controller=ctl,planned=512,completed=len(rr),success=success,unknown=unknown,rate_lower=success/512,rate_upper=(success+unknown)/512,qualified=success/512>=.95,collisions=sum(r['collision'] for r in rr)))
    assert len(rows)==144;write(HERE/'QUALIFICATION.json',rows)
    selection=read(HERE/'HISTORY_SELECTION.json')
    for scene in NEW:
        train=read(V2/'LEARNING_SPLIT_lit_narrow_passage.json')['training_controllers'];q={r['controller'] for r in rows if r['scene']==scene and r['qualified']}
        # New scenes have no pre-existing split of controller qualification: preserve
        # v2 train pool, then leave unavailable types as protocol gaps.
        ctls={t:next((c for c in train if c['type_code']==t and c['controller'] in q),None) for t in sorted({c['type_code'] for c in train})}
        split=read(HERE/f'LEARNING_SPLIT_{scene}.json');selection[scene]=dict(training_states=split['training'],evaluation_states=split['evaluation'],training_controllers=ctls,qualified=sorted(q),missing_types=[t for t,c in ctls.items() if c is None])
    write(HERE/'SCREEN_SELECTION.json',selection);print('Qualification', {s:sum(r['qualified'] for r in rows if r['scene']==s) for s in NEW},flush=True)

def build_fixed(which):
    scenes=OLD if which=='old' else NEW[:2];sel=read(HERE/'SCREEN_SELECTION.json');rows=[]
    for scene in scenes:
        for ctl in sel[scene]['training_controllers'].values():
            if ctl is None:continue
            for s in sel[scene]['training_states']:
                for i in range(16):rows.append(make(scene,ctl,s,action(i)))
    register(which+'_fixed_train',rows)

def rank(stage,scene):
    cases=[c for c in read(HERE/f'{stage}_CASES.json') if c['state']['scene']==scene];groups=defaultdict(list)
    for c in cases:groups[(c['requested_action'],c['controller_payload']['nominal']['type_code'])].append(load(stage,c))
    byaction=defaultdict(dict)
    for (a,t),rr in groups.items():byaction[a][t]=dict(success_lower=sum(r['success'] for r in rr)/len(rr),success_upper=sum(r['success']+r['numerical_failure'] for r in rr)/len(rr),n=len(rr))
    def order(key):
        z=key.replace('eta','').split('_to_');return tuple(map(int,z))
    out=[dict(action=a,type_equal_lower=float(np.mean([r['success_lower'] for r in ts.values()])),type_equal_upper=float(np.mean([r['success_upper'] for r in ts.values()])),per_type=ts) for a,ts in byaction.items()]
    out.sort(key=lambda r:(-r['type_equal_lower'],order(r['action'])));return out

def build_pairs(which):
    scenes=OLD if which=='old' else NEW[:2];sel=read(HERE/'SCREEN_SELECTION.json');rows=[];rankings={}
    for scene in scenes:
        ranking=rank(which+'_fixed_train',scene);rankings[scene]=ranking
        if not ranking:continue
        assert len(ranking)==16
        top=[int(r['action'][3:]) for r in ranking[:3]]
        for a,b in permutations(top,2):
            for ctl in sel[scene]['training_controllers'].values():
                if ctl is None:continue
                for s in sel[scene]['training_states']:rows.append(make(scene,ctl,s,action(a,b)))
    write(HERE/f'{which.upper()}_FIXED_RANKING.json',rankings);register(which+'_pair_train',rows)

def build_eval(which):
    scenes=OLD if which=='old' else NEW[:2];sel=read(HERE/'SCREEN_SELECTION.json');rows=[];chosen={};manifest={r['controller']:r for r in read(HERE/'CONTROLLER_MANIFEST.json')}
    for scene in scenes:
        fixed=rank(which+'_fixed_train',scene);pairs=rank(which+'_pair_train',scene)
        if not fixed or not pairs:chosen[scene]=dict(status='GAP',missing_types=sel[scene]['missing_types']);continue
        assert len(fixed)==16 and len(pairs)==6
        keys=[fixed[0]['action'],pairs[0]['action'],'eta13','eta13_to_10'];keys=list(dict.fromkeys(keys));actions=[]
        for key in keys:
            z=key.replace('eta','').split('_to_');actions.append(action(*map(int,z)))
        chosen[scene]=dict(status='SELECTED_WITH_TYPE_GAPS' if sel[scene]['missing_types'] else 'SELECTED',best_fixed=fixed[0]['action'],best_pair=pairs[0]['action'],fixed_ranking=fixed,pair_ranking=pairs,missing_types=sel[scene]['missing_types'],selection_uses_evaluation_labels=False,bootstrap_reselects=False)
        for ctl in sel[scene]['qualified']:
            for s in sel[scene]['evaluation_states']:
                for a in actions:rows.append(make(scene,manifest[ctl],s,a))
    path=HERE/f'{which.upper()}_FROZEN_CHOICES.json';assert not path.exists();write(path,dict(utc=utc(),scenes=chosen,before_evaluation_preflight=True));seal(f'{which.upper()}_FROZEN_CHOICES.receipt.json',[path],before_first_evaluation=True)
    register(which+'_eval',rows)

def build_main():
    sel=read(HERE/'SCREEN_SELECTION.json');rows=[]
    for scene in NEW:
        for ctl in read(HERE/'CONTROLLER_MANIFEST.json'):
            actions=[action('safety')]
            if ctl['controller'] in sel[scene]['qualified']:actions.extend([action(10),action(13),action(13,10)])
            for s in read(HERE/f'FLOW_STATES_{scene}.json'):
                for a in actions:
                    for seed in (0,1):rows.append(make(scene,ctl,s,a,seed))
    register('swap_main',rows)

def build_moving_eval():
    chosen=read(HERE/'NEW_FROZEN_CHOICES.json')['scenes']['lit_swap8_free'];rows=[];sel=read(HERE/'SCREEN_SELECTION.json')['lit_swap8_moving'];manifest={r['controller']:r for r in read(HERE/'CONTROLLER_MANIFEST.json')}
    if chosen['status']!='GAP':
        for ctl in sel['qualified']:
            for s in sel['evaluation_states']:
                for key in dict.fromkeys([chosen['best_fixed'],chosen['best_pair'],'eta13','eta13_to_10']):
                    z=key.replace('eta','').split('_to_');rows.append(make('lit_swap8_moving',manifest[ctl],s,action(*map(int,z))))
    write(HERE/'MOVING_FROZEN_CHOICES.json',dict(utc=utc(),free_choices_sha256=sha(HERE/'NEW_FROZEN_CHOICES.json'),selected_from_free=True,no_moving_screen=True,choice=chosen));register('moving_eval',rows)
if __name__=='__main__':
    op=sys.argv[1]
    if op=='qualify':qualify()
    elif op=='fixed':build_fixed(sys.argv[2])
    elif op=='pairs':build_pairs(sys.argv[2])
    elif op=='eval':build_eval(sys.argv[2])
    elif op=='main':build_main()
    elif op=='moving':build_moving_eval()
