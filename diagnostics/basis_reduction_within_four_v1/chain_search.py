"""Same Oracle requests, cached/preflighted once and executed in case chains."""
from dispatch import *
from evidence import run_requests
from chain_logic import consume_known
from execution_control import PriorityPause
from budget import used


def run(state,path,request,combo,cap,ceiling):
    gm={tuple(g['identity']):g for g in state['groups']}
    state.setdefault('chain_runs',[])
    while True:
        active=[g for g in state['groups']if not g['success']and not g['gap']and g['searched']<cap]
        if not active:break
        prior=[r for r in state['chain_runs']if r['cap']==cap and not r.get('intentional_ceiling_pause')]
        submissions=sum(len(read(HERE/f'{r["stage"]}_jobs.json'))for r in prior if(HERE/f'{r["stage"]}_jobs.json').exists())
        if submissions>=4:
            for g in active:g['gap']='chain_infrastructure_incomplete_after_three_retries'
            dump(path,state);break
        number=len(state['chain_runs'])+1
        stage=f'{state["code"]}_{state["scope"]}_chain_c{cap:02d}_r{number:03d}'
        items={tuple(g['identity']):[dict(index=k,query_uid=(c:=request(g,k))['rollout_uid'],request=c)
                                    for k in range(g['searched']+1,cap+1)]for g in active}
        potential=[item['request']for group in items.values()for item in group]
        known,_=run_requests(potential,stage+'_potential',eligible=combo.get('no_passage_preserves_four',False),
                             ceiling=ceiling,resolve_only=True)
        labels=[]
        def lookup(uid):
            if uid in known:return known[uid]
            raw=HERE/'raw'/f'{uid}.json'
            return read(raw)if raw.exists()else None
        last=[0.]
        def collect():
            changed=False
            for identity,entries in items.items():
                added=consume_known(gm[identity],entries,lookup)
                labels.extend(dict(identity=list(identity),**q)for q in added)
                changed=changed or bool(added)
            if changed:dump(path,state)
            if time.monotonic()-last[0]>30:
                complete=sum(gm[identity]['success']or gm[identity]['searched']>=cap for identity in items)
                status(stage+'_eta_chains',complete,len(items),f'每案例按冻结顺序连续搜索，成功即停；最多{cap}组。原数值执行器和UID不变；所有潜在请求已官方preflight，未执行项不当失败。')
                last[0]=time.monotonic()
        collect()
        active=[g for g in active if not g['success']and not g['gap']and g['searched']<cap]
        if not active:
            if labels:
                dump(HERE/f'{stage}_labels.json',labels)
                state['completed_stages'].append(stage+'_cache_only');dump(path,state)
            break
        if used()>=ceiling:raise PriorityPause('Chain priority allowance exhausted; unexecuted candidates remain unknown')
        groups=[];drivers={}
        for g in active:
            identity=tuple(g['identity']);remaining=[item for item in items[identity]if item['index']>g['searched']]
            first=remaining[0]['request'];assert lookup(first['rollout_uid'])is None
            driver=dict(first,_chain_stage=stage);drivers[first['rollout_uid']]=driver
            groups.append(dict(identity=list(identity),group_id=digest(dict(code=state['code'],scope=state['scope'],identity=identity)),
                driver_uid=first['rollout_uid'],items=remaining,
                estimated_load=len(remaining)*first['scenario_payload']['config']['max_steps']*first['state']['N']**1.5))
        fast={bool(item['request']['controller_payload'].get('exact_acceleration'))for g in groups for item in g['items']}
        assert len(fast)==1
        dump(HERE/'chain_plans'/f'{stage}.json',dict(groups=groups,known=known,ceiling=ceiling,accelerated=fast.pop(),
             candidate_order_unchanged=True,potential_registration_is_not_execution=True))
        entry=dict(stage=stage,cap=cap,potential_requests=len(potential),drivers=len(drivers),attempts_before=used())
        state['chain_runs'].append(entry);state['chain_pending']=stage;dump(path,state)
        execute(list(drivers.values()),stage,worker_entry='chain_worker.py',max_submissions=4-submissions,progress_callback=collect)
        collect()
        # The existing execute() performs the only official merge after all
        # workers exit. Read back every source actually consumed by the chain.
        with db.connect(True)as con:
            for item in labels:
                r=item['result'];row=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
                assert row and not row['conflict_quarantined']and row['success']==r['success']and row['numerical_failure']==r['numerical_failure']
        dump(HERE/f'{stage}_labels.json',labels)
        with(HERE/'candidates.jsonl').open('a')as stream:
            for item in labels:stream.write(json.dumps(clean(item['result']),ensure_ascii=False)+'\n')
        entry.update(attempts_after=used(),logical_queries=len(labels),
                     completed_groups=sum(gm[k]['success']or gm[k]['searched']>=cap for k in items),
                     intentional_ceiling_pause=used()>=ceiling)
        state['completed_stages'].append(stage);state['chain_pending']=None;dump(path,state)
        if used()>=ceiling and any(not g['success']and not g['gap']and g['searched']<cap for g in state['groups']):
            raise PriorityPause('Chain priority allowance exhausted after preserving all completed results')
    state['complete64']=all(g['success']or g['searched']>=64 for g in state['groups'])
    dump(path,state)
    from analysis_results import write_report
    write_report()
    return state
