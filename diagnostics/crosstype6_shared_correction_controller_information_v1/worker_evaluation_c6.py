"""Evaluate frozen Safety choices first, then reuse only actual certified traces."""
from common import *
from actions_c6 import execute
from reuse_c6 import infer

def run(worker,workers,mode):
    assert mode in ('safety','remaining') and not os.environ.get('XLA_FLAGS')
    for p,h in read(HERE/'evaluation_EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    rows=read(HERE/'evaluation_EXECUTION_CASES.json');p=read(HERE/'evaluation_protocol.json');cache={};completed=0;reused=0
    def safety(c):return not np.any(c['eta_a']) and not np.any(c['eta_b']) and c.get('late_source','given')=='given'
    rows=[c for c in rows if safety(c)==(mode=='safety')]
    # Operational order only: physical state/action/future keys remain sealed.
    order_seed=2026102061+(0 if mode=='safety' else 1)
    order=np.random.default_rng(order_seed).permutation(len(rows));rows=[rows[int(i)] for i in order]
    for i in range(worker,len(rows),workers):
        c=rows[i];rid=c['rollout_uid'];raw=HERE/'evaluation/raw'/f'{rid}.json';alias=HERE/'evaluation/dynamic_aliases'/f'{rid}.json'
        if raw.exists() or alias.exists():continue
        with rollout_db.connect(True) as db:
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(rid,)).fetchone():continue
            reuse=infer(c,p,db) if mode=='remaining' and c.get('c6_schedule') else None
        if reuse:write(alias,reuse);reused+=1;continue
        r=execute(c,p,cache,'evaluation');write(raw,r);cache_writer.append_journal([r],EXP,f'evaluation_{mode}_{worker}');completed+=1
        if (completed+reused)%25==0:print(json.dumps(dict(utc=utc(),worker=worker,mode=mode,physical=completed,local_reuse=reused,index=i)),flush=True)
    if '_frozen_mean' in cache:cache['_frozen_mean'].close()
    write(HERE/f'evaluation_{mode}_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),physical=completed,local_reuse=reused))

if __name__=='__main__':run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
