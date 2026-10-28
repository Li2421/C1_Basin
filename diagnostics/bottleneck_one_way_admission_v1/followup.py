"""Sequential frozen-menu screening and threshold-triggered all-scene tests."""
from dispatch import *
from acceleration import accelerated_case
from evidence import run_requests
from followup_validation import units,certify,no_passage
from budget import used,limit
import copy

def identity(e,ctl,si):return [e['type_id'],e['level'],ctl['type_code'],si]

def original_family():
    f=read(HERE/'FAMILIES_THIS_ROUND.json')['families'][0]
    return dict(scheme='A',name='通道单向放行·三列版',candidate_pool=f['candidate_pool'],coefficients=3)

def search(family,scope,make):
    scheme=family['scheme'];path=HERE/f'FOLLOWUP_{scheme}_{scope}.json'
    entries=read(HERE/'SCENES.json')['entries'];ctls=controllers()
    hard={tuple((g['type_id'],g['level'],g['controller'],g['state_index']))for g in read(HERE/'HARD_CASES.json')['cases']}
    if path.exists():state=read(path)
    else:
        inherited={}
        if scheme=='A':inherited={tuple(g['identity']):g for g in read(HERE/'ORACLE_STATE.json')['groups']if g['family']==0}
        elif scope=='all'and(HERE/f'FOLLOWUP_{scheme}_hard.json').exists():inherited={tuple(g['identity']):g for g in read(HERE/f'FOLLOWUP_{scheme}_hard.json')['groups']}
        groups=[]
        for e in entries:
            for si in range(8):
                for ctl in ctls:
                    key=identity(e,ctl,si)
                    if scope=='hard'and tuple(key)not in hard:continue
                    if tuple(key)in inherited:g=copy.deepcopy(inherited[tuple(key)])
                    else:g=dict(identity=key,scene=e['scene']['name'],hard=tuple(key)in hard,success=False,searched=0,first_success=None,queries=[],gap=None)
                    groups.append(g)
        state=dict(scheme=scheme,scope=scope,groups=groups,pending=None,completed_stages=[],complete=False)
        dump(path,state)
    if state['complete']:return state
    scenes={(e['type_id'],e['level']):e['scene']for e in entries};controls={c['type_code']:c for c in ctls};gm={tuple(g['identity']):g for g in state['groups']}
    def request(g,k):
        t,l,c,si=g['identity']
        return make(scenes[(t,l)],controls[c],si,family['candidate_pool'][k-1],purpose='followup_'+scope)
    def consume():
        pending=state['pending'];cases=[]
        for item in pending['frontier']:
            c=request(gm[tuple(item['identity'])],item['index']);assert c['rollout_uid']==item['query_uid'];cases.append(c)
        rows=run_requests(cases,pending['stage']);labels=[]
        for item in pending['frontier']:
            g=gm[tuple(item['identity'])];r=rows.get(item['query_uid'])
            if r is None:g['gap']='infrastructure_incomplete_after_retries';continue
            assert not g['success']and g['searched']==item['index']-1
            q=dict(index=item['index'],query_uid=item['query_uid'],result=r)
            g['queries'].append(q);g['searched']=item['index']
            if r['success']:g.update(success=True,first_success=item['index'],solution=q)
            labels.append(dict(identity=g['identity'],**q))
        dump(HERE/(pending['stage']+'_labels.json'),labels)
        state['completed_stages'].append(pending['stage']);state['pending']=None;dump(path,state)
        from followup_reporting import write_report
        write_report()
    if state['pending']:consume()
    for k in range(1,65):
        active=[g for g in state['groups']if not g['success']and not g['gap']and g['searched']<k]
        if not active:continue
        frontier=[]
        for g in active:
            assert g['searched']==k-1
            c=request(g,k);frontier.append(dict(identity=g['identity'],index=k,query_uid=c['rollout_uid']))
        state['pending']=dict(stage=f'follow_{scheme}_{scope}_k{k:02d}',frontier=frontier);dump(path,state);consume()
    state['complete']=all(g['success']or g['searched']==64 for g in state['groups']);dump(path,state)
    return state

def fixed(family,make):
    scheme=family['scheme'];path=HERE/f'FOLLOWUP_{scheme}_fixed.json'
    state=read(path)if path.exists()else dict(methods={})
    base=read(HERE/'FAMILIES_THIS_ROUND.json')['fixed_reference']
    for method,sp in [('four_column_fixed',base),('scheme_fixed',family['candidate_pool'][0])]:
        if state['methods'].get(method,{}).get('complete'):continue
        cases=[];records=[]
        for e in read(HERE/'SCENES.json')['entries']:
            for si in range(12):
                for ctl in controllers():
                    c=make(e['scene'],ctl,si,sp,purpose='followup_all_fixed');cases.append(c)
                    records.append(dict(identity=identity(e,ctl,si),query_uid=c['rollout_uid'],spec=sp))
        rows=run_requests(cases,f'follow_{scheme}_fixed_{method}')
        state['methods'][method]=dict(complete=len(rows)==len(cases),cases=len(cases),records=[dict(x,result=rows.get(x['query_uid']))for x in records]);dump(path,state)
    return state

def qualifies(state):
    groups=state['groups'];return bool(groups)and 10*sum(g['success']for g in groups)>=9*len(groups)

def full(family,make):
    from followup_reporting import allscene,write_report
    search(family,'all',make);fixed(family,make);allscene(family);write_report()

def main():
    os.environ['OWA_BUDGET_GROUP']='followup'
    assert read(HERE/'ORIGINAL_COMPLETE.json')['complete'],'Original delivery must be complete before follow-up'
    assert sha(HERE/'followup_runtime.py')==read(HERE/'FOLLOWUP_DEFINITIONS.json')['runtime_sha256']
    assert all(sha(HERE/name)==expected for name,expected in read(HERE/'FOLLOWUP_ENGINE_FREEZE.json')['files'].items()),'Followup engine changed after readiness; review implementation and affected evidence'
    units();families=[original_family()]+read(HERE/'FAMILIES_FOLLOWUP.json')['families']
    path=HERE/'FOLLOWUP_PROGRESS.json'
    progress=read(path)if path.exists()else dict(screened=[],qualifiers=[],full_complete=[],deferred=[],budget_exhausted=False)
    from followup_reporting import write_report,finish
    hard_count=read(HERE/'HARD_CASES.json')['count'];makers={};first_full=None
    try:
        for index,family in enumerate(families):
            scheme=family['scheme']
            if scheme=='A':make=accelerated_case if read(HERE/'ACCELERATION_CERTIFICATE.json')['passed']else case
            else:
                make=certify(family);no_passage(family,make)
            makers[scheme]=make
            state=search(family,'hard',make)
            if scheme not in progress['screened']:progress['screened'].append(scheme)
            if qualifies(state)and scheme not in progress['qualifiers']:progress['qualifiers'].append(scheme)
            dump(path,progress);write_report()
            if scheme not in progress['qualifiers']or scheme in progress['full_complete']:continue
            # First qualifier's full run is highest priority. Reserve a worst-
            # case hard-search allowance for remaining schemes before spending
            # on a second full run, as the prompt's budget priority requires.
            remaining_schemes=len(families)-index-1
            reserve=remaining_schemes*(hard_count*64+120)
            first_full=progress['full_complete'][0]if progress['full_complete']else None
            if first_full and limit()-used()<reserve+8000:
                if scheme not in progress['deferred']:progress['deferred'].append(scheme)
                dump(path,progress);continue
            full(family,make)
            if read(HERE/f'ALLSCENE_{scheme}.json')['complete']:progress['full_complete'].append(scheme)
            dump(path,progress)
        pending=[f for f in families if f['scheme']in progress['qualifiers']and f['scheme']not in progress['full_complete']]
        pending.sort(key=lambda f:(f['coefficients'],-sum(g['success']for g in read(HERE/f"FOLLOWUP_{f['scheme']}_hard.json")['groups'])))
        for family in pending:
            scheme=family['scheme'];full(family,makers[scheme])
            if read(HERE/f'ALLSCENE_{scheme}.json')['complete']:progress['full_complete'].append(scheme)
            dump(path,progress)
    except RuntimeError as exc:
        if 'actual new rollout budget exhausted'not in str(exc):raise
        progress['budget_exhausted']=True;progress['budget_error']=str(exc);dump(path,progress)
    write_report();finish(progress)

if __name__=='__main__':main()
