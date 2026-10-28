"""Resumable admission study: certification, fixed settings, hard cases, regression."""
from dispatch import *
from acceleration import accelerated_case
from validation import certify
from evidence import run_requests,no_passage_certificate
import traceback

def key(e,ctl,si):return (e['type_id'],e['level'],ctl['type_code'],si)

def main():
    from reporting import write_report
    # Only one coordinator may merge. Previous independently submitted workers
    # must finish before recovery reads or consumes their results.
    own=os.environ.get('SLURM_JOB_ID')
    while any(j['name'].startswith('owa1_')and j['job_id']!=own for j in jobs()):time.sleep(10)
    if list((HERE/'raw').glob('*.json')):merge('resume_recovery',[dict(rollout_uid=p.stem)for p in(HERE/'raw').glob('*.json')])
    assert (HERE/'HANDOFF_COMPLETE.json').exists()
    acceleration=certify();make=accelerated_case if acceleration else case
    no_passage_certificate(make);write_report()
    selected={x['scene']for x in read(HERE/'REPRESENTATIVE17_FROZEN.json')['selected']}
    scenes=[e for e in read(HERE/'SCENES.json')['entries']if e['scene']['name']in selected]
    controls=controllers();family=read(HERE/'FAMILIES_THIS_ROUND.json')
    fixed_path=HERE/'FIXED_RESULTS.json';fixed=read(fixed_path)if fixed_path.exists()else dict(methods={})
    for method,sp in [('four_column_fixed',family['fixed_reference']),('admission_fixed',family['fixed_admission'])]:
        if fixed['methods'].get(method,{}).get('complete'):continue
        cases=[];identities=[]
        for e in scenes:
            for si in range(12):
                for ctl in controls:
                    c=make(e['scene'],ctl,si,sp,purpose='fixed_parameter_evaluation');cases.append(c)
                    identities.append(dict(identity=list(key(e,ctl,si)),query_uid=c['rollout_uid'],spec=sp))
        rows=run_requests(cases,'fixed_'+method)
        records=[dict(x,result=rows.get(x['query_uid']))for x in identities]
        fixed['methods'][method]=dict(complete=len(rows)==len(cases),cases=len(cases),records=records)
        dump(fixed_path,fixed);write_report()
    hard={tuple((g['type_id'],g['level'],g['controller'],g['state_index']))for g in read(HERE/'HARD_CASES.json')['cases']}
    path=HERE/'ORACLE_STATE.json'
    if path.exists():state=read(path)
    else:
        groups=[]
        for fi in range(4):
            for e in scenes:
                for si in range(8):
                    for ctl in controls:
                        identity=key(e,ctl,si)
                        if fi and identity not in hard:continue
                        groups.append(dict(family=fi,identity=list(identity),hard=identity in hard,scene=e['scene']['name'],success=False,searched=0,first_success=None,queries=[],gap=None))
        state=dict(groups=groups,pending=None,completed_stages=[])
        dump(path,state)
    gm={(g['family'],*g['identity']):g for g in state['groups']}
    scene_map={(e['type_id'],e['level']):e['scene']for e in scenes};ctl_map={c['type_code']:c for c in controls}
    def request(g,k):
        t,l,ctl,si=g['identity'];sp=family['families'][g['family']]['candidate_pool'][k-1]
        return make(scene_map[(t,l)],ctl_map[ctl],si,sp,purpose='admission_oracle')
    def finish_pending():
        pending=state['pending'];cases=[]
        for item in pending['frontier']:
            g=gm[tuple(item['group'])];c=request(g,item['index']);assert c['rollout_uid']==item['query_uid'];cases.append(c)
        rows=run_requests(cases,pending['stage']);labels=[]
        for item in pending['frontier']:
            g=gm[tuple(item['group'])];r=rows.get(item['query_uid'])
            if r is None:g['gap']='infrastructure_incomplete_after_retries';continue
            assert not g['success']and g['searched']==item['index']-1
            q=dict(index=item['index'],query_uid=item['query_uid'],result=r)
            g['queries'].append(q);g['searched']=item['index']
            if r['success']:g.update(success=True,first_success=item['index'],solution=q)
            labels.append(dict(identity=g['identity'],family=g['family'],**q))
        dump(HERE/f"{pending['stage']}_labels.json",labels)
        state['completed_stages'].append(pending['stage']);state['pending']=None;dump(path,state);write_report()
    if state['pending']:finish_pending()
    phases=[('hard_main',0,True,64),('hard_slow',1,True,16),('hard_only',2,True,16),('hard_role',3,True,16),('regression',0,False,16)]
    for phase,fi,ishard,cap in phases:
        for k in range(1,cap+1):
            stage=f'{phase}_k{k:02d}'
            if stage in state['completed_stages']:continue
            active=[g for g in state['groups']if g['family']==fi and g['hard']==ishard and not g['success']and not g['gap']and g['searched']<k]
            if not active:continue
            frontier=[]
            for g in active:
                assert g['searched']==k-1
                c=request(g,k);frontier.append(dict(group=[fi,*g['identity']],index=k,query_uid=c['rollout_uid']))
            state['pending']=dict(stage=stage,frontier=frontier);dump(path,state);finish_pending()
        write_report()
    from reporting import finalize
    finalize()
    from finish_extras import main as finish_extras
    finish_extras()
    dump(HERE/'ORIGINAL_COMPLETE.json',read(HERE/'COMPLETE.json'))
    if(HERE/'FAMILIES_FOLLOWUP.json').exists():
        dump(HERE/'COMPLETE.json',dict(complete=False,original_complete=True,followup_pending=True))
        while not(HERE/'FOLLOWUP_READY').exists():
            status('等待续接方案执行器',0,6,'原说明已交付；续接方案定义已冻结，等待实现与零rollout检查完成。')
            time.sleep(30)
        from followup import main as followup_main
        followup_main()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(HERE/'PIPELINE_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        from reporting import write_report
        write_report()
        with(HERE/'GAPS.md').open('a')as f:f.write('\n流程中断并保留可恢复清单：'+repr(exc)+'\n')
        raise
