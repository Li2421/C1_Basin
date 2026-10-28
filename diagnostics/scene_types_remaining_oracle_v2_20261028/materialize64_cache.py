"""Close search groups fully answered by existing evidence; zero new rollout."""
import json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def main():
    groups=read(HERE/'ORACLE64_GROUPS.json')['groups'];count=0
    for g in groups:
        if any(x['source']=='new_missing'for x in g['candidates']):continue
        target=HERE/'oracle64_groups'/f"{g['family']}_{g['type_id']}_{g['controller']}_{g['state_index']}.json"
        if target.exists():continue
        result=dict(family=g['family'],type_id=g['type_id'],controller=g['controller'],state_index=g['state_index'],first_success=None,visited=[],actual_new_rollouts=0)
        for p in g['candidates']:
            r=read(p['path']);assert r['rollout_uid']==p['uid'] and r['spec']==p['spec']
            result['visited'].append(dict(candidate=p['candidate'],uid=p['uid'],outcome=r['outcome'],source=p['source']))
            if r['success']:
                result['first_success']=p['candidate'];result['success_uid']=p['uid'];break
        assert result['first_success'] is not None or len(result['visited'])==64
        result['search_count']=result['first_success'] or 64
        target.parent.mkdir(exist_ok=True)
        tmp=target.with_suffix(f'.cache-{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(target);count+=1
    print('Reused complete cached search groups',count,'new rollouts0')
if __name__=='__main__':main()
