"""Exercise coordinator cache and worker-result paths using archived labels."""
from pathlib import Path
import ast,copy,hashlib,json,sys,time,types
from chain_logic import consume_known,walk

HERE=Path(__file__).resolve().parent


def main():
    source=json.loads((HERE/'SEARCH_level3_01_screen.json').read_text())
    late=next(g for g in source['groups']if g['success']and g['first_success']==60)
    failed=next(g for g in source['groups']if not g['success']and g['searched']==64)
    original=[late,failed];records={q['query_uid']:q['result']for g in original for q in g['queries']}
    ids={(tuple(g['identity']),q['index']):q['query_uid']for g in original for q in g['queries']}
    result_rows={r['rollout_uid']:r for r in records.values()}
    class Cursor:
        def __init__(self,row):self.row=row
        def fetchone(self):return self.row
    class Connection:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def execute(self,sql,args):
            r=result_rows[args[0]]
            return Cursor(dict(success=r['success'],numerical_failure=r['numerical_failure'],conflict_quarantined=0))
    tree=ast.parse((HERE/'chain_search.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    previous=sys.modules.get('analysis_results')
    sys.modules['analysis_results']=types.SimpleNamespace(write_report=lambda:None)
    checks=[]
    try:
        for cached in (True,False):
            work=HERE/'unit_chain_fixture'/('cached'if cached else'worker_results');work.mkdir(parents=True,exist_ok=True)
            groups=[]
            for original_group in original:
                g=copy.deepcopy(original_group);g.update(success=False,searched=16,first_success=None,gap=None)
                g['queries']=g['queries'][:16];g.pop('solution',None);groups.append(g)
            state=dict(code='unit_only',scope='screen',groups=groups,completed_stages=[],pending=None)
            def dump(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
            def read(path):return json.loads(path.read_text())
            def request(g,k):
                uid=ids.get((tuple(g['identity']),k),f'unreachable_{k}')
                return dict(rollout_uid=uid,controller_payload=dict(exact_acceleration=True),
                            scenario_payload=dict(config=dict(max_steps=100)),state=dict(N=2))
            def resolve(cases,stage,**kwargs):
                assert kwargs['resolve_only']
                available={c['rollout_uid']:records[c['rollout_uid']]for c in cases
                           if c['rollout_uid']in records and(cached or not records[c['rollout_uid']]['success']and int(next(k for (identity,k),v in ids.items()if v==c['rollout_uid']))<=18)}
                return available,[c for c in cases if c['rollout_uid']not in available]
            def execute(drivers,stage,**kwargs):
                assert not cached,'Cached evidence launched a worker'
                plan=read(work/'chain_plans'/f'{stage}.json')
                for group in plan['groups']:
                    def evaluate(item):
                        r=records[item['query_uid']]
                        dump(work/'raw'/f'{item["query_uid"]}.json',r)
                        return r
                    walk(group['items'],evaluate,lambda item,r:kwargs['progress_callback']())
                dump(work/f'{stage}_jobs.json',[dict(retry=0)])
            ns=dict(HERE=work,read=read,dump=dump,run_requests=resolve,consume_known=consume_known,
                    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest(),
                    status=lambda *args,**kwargs:None,time=time,json=json,used=lambda:0,clean=lambda x:x,
                    execute=execute,db=types.SimpleNamespace(connect=lambda *args:Connection()),PriorityPause=RuntimeError)
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(HERE/'chain_search.py'),'exec'),ns)
            ns['run'](state,work/'STATE.json',request,{},64,80000)
            for actual,expected in zip(state['groups'],original):
                for key in ('success','searched','first_success'):assert actual[key]==expected[key]
            assert state['complete64']
            checks.append(dict(cache_only=cached,passed=True,groups=2,new_rollouts=0))
    finally:
        if previous is None:sys.modules.pop('analysis_results',None)
        else:sys.modules['analysis_results']=previous
    output=dict(passed=True,checks=checks,new_rollouts=0,scope='Coordinator-only fixtures; no simulator, official database or worker process invoked.')
    (HERE/'CHAIN_COORDINATOR_CHECK.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
