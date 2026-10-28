"""One-CPU sequential searches: at most64 settings per failed context."""
import os
os.environ.pop('STB_COORDINATOR',None)
from core import *
from acceleration import enable
from recording_fix import install
from runner import work
import time,sys

def main(stage,index):
    enable();install()
    queue=read(HERE/'queues'/f'{stage}.json')
    for g in queue['shards'][index]:
        result=dict(family=g['family'],type_id=g['type_id'],controller=g['controller'],state_index=g['state_index'],first_success=None,visited=[],actual_new_rollouts=0)
        for point in g['candidates']:
            p=Path(point['path'])
            if p.exists():r=read(p)
            else:
                c=queue['requests'][point['uid']]
                assert c['rollout_uid']==point['uid']
                # Requests are serial inside a context; no extra setting runs
                # after its first successful candidate.
                attempt=HERE/'attempts'/f"{c['rollout_uid']}_{time.time_ns()}.json"
                dump(attempt,dict(rollout_uid=c['rollout_uid'],job=os.environ.get('SLURM_JOB_ID'),time=time.time(),phase='oracle64',family=g['family'],candidate=point['candidate']))
                if len(list((HERE/'attempts').glob('*.json')))>20000:raise RuntimeError('oracle64 finite attempt cap')
                r=work(c);result['actual_new_rollouts']+=1
            result['visited'].append(dict(candidate=point['candidate'],uid=point['uid'],outcome=r['outcome'],source=point['source']))
            if r['success']:
                result['first_success']=point['candidate'];result['success_uid']=point['uid'];break
        assert result['first_success'] is not None or len(result['visited'])==64
        result['search_count']=result['first_success'] or 64
        dump(HERE/'oracle64_groups'/f"{g['family']}_{g['type_id']}_{g['controller']}_{g['state_index']}.json",result)
        print('family',g['family'],'type',g['type_id'],g['controller'],'state',g['state_index'],'first',result['first_success'],'new',result['actual_new_rollouts'],flush=True)

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
