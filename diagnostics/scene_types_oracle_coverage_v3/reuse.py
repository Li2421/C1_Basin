"""Explicit horizon transfer; never forge an exact cache entry for a new UID."""
from dispatch import *
from history import index,request_key
from functools import lru_cache

@lru_cache(maxsize=1)
def manifest():
    p=HERE/'REUSE_MANIFEST.json'
    return read(p)if p.exists()else dict(records={},policy='Certified v1 horizon transfer, distinct from exact official cache reuse.')

def cached(c,stage,con):
    """Consume only reached evidence; None means this exact candidate needs work."""
    uid=c['rollout_uid'];path=HERE/'raw'/f'{uid}.json'
    row=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
    if row:
        assert not row['conflict_quarantined']
        if path.exists():r=read(path)
        else:
            source=con.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf ON rs.source_uid=sf.source_uid WHERE rs.rollout_uid=? LIMIT 1',(uid,)).fetchone()
            assert source
            with Path(source['path']).open()as f:r=next(json.loads(line)for i,line in enumerate(f,1)if i==source['source_line'])
        assert r['rollout_uid']==uid and r['success']==row['success']and r['numerical_failure']==row['numerical_failure']
        return dict(r,source='official_exact_cache')
    assert not path.exists(),('Unmerged raw result before new candidate lookup',uid)
    if not read(HERE/'PREFIX_CHECK.json')['passed']:return None
    old=index().get(request_key(c))
    if old is None:return None
    item=dict(v1_rollout_uid=old['rollout_uid'],v1_path=old['path'],target_query_uid=uid,
              target_scenario_uid=c['scenario_uid'],target_state_uid=c['state_uid'],
              target_controller_uid=c['controller_uid'],stage=stage)
    if old['timeout']:
        item['action']='old_timeout_extend_only_when_candidate_reached';manifest()['records'][uid]=item
        return None
    row=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(old['rollout_uid'],)).fetchone()
    r=read(old['path'])
    assert row and not row['conflict_quarantined']and row['success']==r['success']and row['numerical_failure']==r['numerical_failure']
    item.update(action='reuse_terminal_outcome_after_prefix_certificate',outcome=r['outcome'],prefix_certificate_sha256=sha(HERE/'PREFIX_CHECK.json'))
    manifest()['records'][uid]=item
    r=dict(r,source='v1_prefix_certified_horizon_transfer',logical_query_uid=uid,
           horizon_time=c['scenario_payload']['config']['max_steps']*c['scenario_payload']['config']['dt'])
    r['truncated_time']=r['completion_time']if r['success']else r['horizon_time']
    return r

def flush():dump(HERE/'REUSE_MANIFEST.json',manifest())

def requests(cases,stage):
    enabled=read(HERE/'PREFIX_CHECK.json')['passed']
    history=index()if enabled else{}
    result={};actual=[];borrowed=[]
    m=manifest()
    with db.connect(True)as con:
        for c in cases:
            uid=c['rollout_uid'];old=history.get(request_key(c))
            if old and not old['timeout']:
                row=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(old['rollout_uid'],)).fetchone()
                assert row and not row['conflict_quarantined']
                r=read(old['path']);assert row['success']==r['success']and row['numerical_failure']==r['numerical_failure']
                r=dict(r,source='v1_prefix_certified_horizon_transfer',logical_query_uid=uid,
                       horizon_time=c['scenario_payload']['config']['max_steps']*c['scenario_payload']['config']['dt'])
                r['truncated_time']=r['completion_time']if r['success']else r['horizon_time']
                result[uid]=r
                item=dict(v1_rollout_uid=old['rollout_uid'],v1_path=old['path'],target_query_uid=uid,
                          target_scenario_uid=c['scenario_uid'],target_controller_uid=c['controller_uid'],
                          action='reuse_terminal_outcome_after_prefix_certificate',outcome=r['outcome'],
                          prefix_certificate_sha256=sha(HERE/'PREFIX_CHECK.json'),stage=stage)
                if uid in m['records']:assert m['records'][uid]['v1_rollout_uid']==item['v1_rollout_uid']
                m['records'][uid]=item;borrowed.append(item)
            else:
                actual.append(c)
                if old:
                    m['records'][uid]=dict(v1_rollout_uid=old['rollout_uid'],v1_path=old['path'],target_query_uid=uid,
                         action='old_timeout_extend_only_when_candidate_reached',stage=stage)
    dump(HERE/f'{stage}_reuse.json',dict(borrowed=borrowed,actual_or_exact_cache_uids=[c['rollout_uid']for c in actual]))
    dump(HERE/'REUSE_MANIFEST.json',m)
    if actual:
        rows=execute(actual,stage)
        result.update({r['rollout_uid']:r for r in rows})
    else:
        dump(HERE/f'{stage}_execution.json',dict(complete=True,requests=0,new=0,borrowed=len(borrowed),missing=[],attempts=[]))
    return result
