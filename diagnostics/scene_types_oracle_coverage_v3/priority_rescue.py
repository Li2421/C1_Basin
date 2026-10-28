"""Prioritize frozen four-column failures; resume the main search afterwards.

This changes scheduling only. Local search state is separate; all rollouts use
the same frozen candidate identities and the official preflight/merger path.
"""
import copy
from collections import Counter
from dispatch import HERE, read, dump, db, status
from reuse import cached, flush, requests
from representative_scope import in_requested_scope

ORDER = [5, 6, 1]

def identity(g):
    return (g['type_id'], g['level'], g['controller'], g['state_index'])

def witnesses(state):
    result = {tuple(x['identity']): x for x in state['initial_witnesses']}
    for g in state['groups']:
        if g['success'] and identity(g) not in result:
            result[identity(g)] = dict(identity=list(identity(g)), family=g['family'],
                                      first_success=g['first_success'], solution=g['solution'],
                                      source='priority_rescue_search')
    return result

def summarize(state):
    known = witnesses(state)
    by_type = []
    for t in sorted({x['type_id'] for x in state['targets']}):
        rows = [g for g in state['targets'] if g['type_id'] == t]
        by_type.append(dict(type_id=t, four_column_failures=len(rows),
                            rescued=sum(identity(g) in known for g in rows)))
    result = dict(complete=state.get('complete', False), targets=len(state['targets']),
                  already_rescued=len(state['initial_witnesses']), rescued=len(known),
                  newly_rescued=len(known)-len(state['initial_witnesses']),
                  unresolved=len(state['targets'])-len(known), by_type=by_type,
                  witnesses=list(known.values()), family_order=ORDER,
                  outcome_counts=dict(Counter(q['outcome'] for g in state['groups'] for q in g['queries'])),
                  note='Targeted development search. Any-family witness stops this case across priority families. No continuous-space impossibility claim; global family coverage is reported separately.')
    dump(HERE/'PRIORITY_RESCUE_SUMMARY.json',result)
    (HERE/'PRIORITY_RESCUE_REPORT.md').write_text(
        '# 四列失败案例优先补救\n\n'
        f"四列失败 {result['targets']} 案例；旧证据已补救 {result['already_rescued']}；"
        f"本阶段新增补救 {result['newly_rescued']}；当前仍未找到 {result['unresolved']}。\n\n"
        '优先顺序为五列[G,R,T1,C2,Y1]、两列[G,C2]、单列[T5]，按候选序号交替推进。'
        '每个案例只要任一族找到解就停止本阶段搜索。三列[R,T1,T5]只复用当前已完成证据，优先阶段结束后继续其原搜索。'
        '冻结菜单内最多64候选/族；精确缓存结果不重跑。不能将本失败子集用于总体成功率排名。\n\n'
        '| 场景类型 | 四列失败 | 当前已补救 |\n|---|---:|---:|\n'+
        ''.join(f"| {r['type_id']} | {r['four_column_failures']} | {r['rescued']} |\n" for r in by_type))
    return result

def run(groups, make, scene_map, ctl_map, menus, consume):
    path=HERE/'PRIORITY_RESCUE_STATE.json'
    if path.exists():
        state=read(path)
    else:
        targets=[g for g in groups if g['family']==4 and in_requested_scope(dict(g,family=2))
                 and not g['success'] and g['searched']==64]
        ids={identity(g) for g in targets}
        known={}
        for g in groups:
            if identity(g) in ids and g['success'] and identity(g) not in known:
                known[identity(g)]=dict(identity=list(identity(g)),family=g['family'],
                    first_success=g['first_success'],solution=g['solution'],source='existing_main_search')
        state=dict(targets=copy.deepcopy(targets),initial_witnesses=list(known.values()),
            groups=copy.deepcopy([g for g in groups if g['family'] in ORDER and identity(g) in ids]),
            family_order=ORDER,pending=None,completed_stages=[],complete=False,
            history_sources=['SEARCH_STATE.json','FAMILIES_64.json','REPRESENTATIVE17_FROZEN.json',
                             'REUSE_MANIFEST.json','shared_rollout_db official exact cache'],
            main_search_preserved=True)
        dump(path,state)
    if state['complete']:
        summarize(state);return
    gm={(g['family'],*identity(g)):g for g in state['groups']}
    def make_case(g,ki):
        return make(scene_map[(g['type_id'],g['level'])]['scene'],ctl_map[g['controller']],
                    g['state_index'],menus[g['family']]['candidate_pool'][ki-1],purpose='v3_oracle_coverage')
    def finish_pending():
        pending=state['pending'];stage=pending['stage'];frontier=[]
        for item in pending['frontier']:
            g=gm[tuple(item['group'])];c=make_case(g,item['candidate_index'])
            assert c['rollout_uid']==item['target_query_uid']
            frontier.append((g,c,item['candidate_index']))
        rows=requests([c for g,c,k in frontier],stage)
        labels=[]
        for g,c,ki in frontier:
            if c['rollout_uid'] not in rows:
                raise RuntimeError('Priority rescue infrastructure result missing: '+c['rollout_uid'])
            labels.append(consume(g,c,rows[c['rollout_uid']],ki))
        dump(HERE/f'{stage}_labels.json',labels)
        state['completed_stages'].append(stage);state['pending']=None;dump(path,state)
        summarize(state)
    if state['pending']:finish_pending()
    for k in range(1,65):
        for fi in ORDER:
            stage=f'rescue_f{fi}_k{k:02d}'
            if stage in state['completed_stages']:continue
            known=witnesses(state);frontier=[];cache_labels=[]
            with db.connect(True)as con:
                for g in state['groups']:
                    if g['family']!=fi or identity(g) in known or g.get('gap') or g.get('skipped') or g['searched']>=k:
                        continue
                    # No later candidate may be inspected before its prefix.
                    assert g['searched']==k-1
                    c=make_case(g,k);r=cached(c,stage,con)
                    if r is None:
                        frontier.append(dict(group=[fi,*identity(g)],candidate_index=k,target_query_uid=c['rollout_uid']))
                    else:
                        cache_labels.append(consume(g,c,r,k))
            flush()
            dump(HERE/f'{stage}_cache_labels.json',cache_labels)
            if frontier:
                state['pending']=dict(stage=stage,frontier=frontier);dump(path,state)
                finish_pending()
            else:
                state['completed_stages'].append(stage);dump(path,state)
            summary=summarize(state)
            status('四列失败案例优先补救',summary['rescued'],summary['targets'],
                   f"候选族 {fi}，候选序号 {k}/64；已有补救 {summary['already_rescued']}，新增 {summary['newly_rescued']}。三列主搜索暂停。")
            if summary['unresolved']==0:break
        if summarize(state)['unresolved']==0:break
    state['complete']=True;dump(path,state);summarize(state)

