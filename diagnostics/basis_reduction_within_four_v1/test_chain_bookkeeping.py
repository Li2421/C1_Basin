"""Replay archived Oracle labels through the new scheduling bookkeeping only."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
from chain_logic import consume_known,walk

HERE=Path(__file__).resolve().parent


def main():
    tests=0;datasets=[]
    for path in sorted(HERE.glob('SEARCH_*_screen.json')):
        state=json.loads(path.read_text());groups=state['groups']
        if len(groups)!=272 or not all(g['success']or g['searched']==64 for g in groups):continue
        datasets.append(path.name)
        for original in groups:
            for prefix in (0,8,12,16,32,48):
                old_queries=original['queries']
                before=[copy.deepcopy(q)for q in old_queries if q['index']<=prefix]
                group=dict(identity=original['identity'],success=bool(before and before[-1]['result']['success']),
                           searched=len(before),first_success=None,queries=before)
                if group['success']:group.update(first_success=before[-1]['index'],solution=before[-1])
                known={q['query_uid']:q['result']for q in old_queries}
                indices={q['index']:q['query_uid']for q in old_queries}
                items=[dict(index=k,query_uid=indices.get(k,f'unreachable_after_success_{k}'))for k in range(group['searched']+1,65)]
                evaluated=[]
                def evaluate(item):
                    evaluated.append(item['index'])
                    assert item['query_uid']in known,'A candidate after first success was evaluated'
                    return known[item['query_uid']]
                returned=[]
                if not group['success']:walk(items,evaluate,lambda item,r:returned.append((item,r)))
                # Poll once with a partial cache, then resume with the complete
                # cache, as after a coordinator/worker infrastructure restart.
                split=len(returned)//2
                partial={item['query_uid']:r for item,r in returned[:split]}
                consume_known(group,items,partial.get)
                consume_known(group,items,known.get)
                for key in ('success','searched','first_success'):
                    assert group[key]==original[key],(path,original['identity'],prefix,key)
                assert len(group['queries'])==group['searched']
                assert [q['index']for q in group['queries']]==list(range(1,group['searched']+1))
                assert evaluated==list(range(min(prefix,original['searched'])+1,original['searched']+1))
                tests+=1
    # Unknown is a recorded non-success, and an unexecuted request is unknown.
    items=[dict(index=k,query_uid=str(k))for k in (1,2,3)]
    values={'1':dict(success=0,numerical_failure=1),'2':dict(success=1,numerical_failure=0)}
    seen=[];walk(items,lambda item:values[item['query_uid']],lambda item,r:seen.append(item['index']))
    assert seen==[1,2]
    group=dict(success=False,searched=0,first_success=None,queries=[])
    consume_known(group,items,{'1':values['1']}.get)
    assert group['searched']==1 and not group['success']
    freeze=json.loads((HERE/'DESIGN_FREEZE.json').read_text())
    unchanged=all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected for name,expected in freeze['numerical_files'].items())
    assert unchanged
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),passed=True,
        archived_state_prefix_replays=tests,datasets=datasets,extra_unknown_and_stop_checks=2,
        numerical_sources_unchanged=len(freeze['numerical_files']),new_rollouts=0,
        scope='Bookkeeping/scheduling contract only. The numerical runner, nominal, solver, scene, seed, coefficient pool and per-case candidate order remain unchanged.')
    (HERE/'CHAIN_BOOKKEEPING_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
