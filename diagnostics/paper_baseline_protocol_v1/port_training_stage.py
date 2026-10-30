"""Matched return-handling comparison, training scenes only."""
from core import *
from dispatch import execute,status
from execution_policy import active_methods
from freeze_menus import DEFAULTS
from scene_catalog import cohort
from request_factory import paper
from port_gates import smoke,unhandled
import datetime

def before_request(request):
    request=json.loads(json.dumps(request));method=request['controller_payload']['paper_method']
    request['controller_payload']['paper_method_adapter']=str(HERE/'inputs/diagnostics/scene_types_oracle_coverage_v3/paper_methods'/method/'adapter.py')
    request['controller_uid']=db.uid('ctl',dict(scenario=request['scenario_uid'],payload=request['controller_payload']))
    request['rollout_uid']=db.uid('roll',dict(state=request['state_uid'],eta=request['eta_uid'],controller=request['controller_uid'],seed=request['seed_key']))
    return request

def run():
    path=HERE/'PORT_FIX_BEFORE_AFTER.json'
    if path.exists()and read(path).get('complete'):return read(path)
    smoke()
    groups=[]
    for method in active_methods():
        for benchmark in ('A','B'):
            requests=[];owners=[]
            for scene in cohort(benchmark,'train'):
                for robot in range(scene['states'][0]['N']):
                    after=paper(benchmark,scene,0,method,DEFAULTS[method],solo=robot,phase='applicability',purpose='matched_training_solo_return_repair')
                    for label,request in [('before',before_request(after)),('after',after)]:
                        requests.append(request);owners.append(dict(scene=scene['name'],category=scene['factor']if benchmark=='A'else scene['type_id'],robot=robot,variant=label,rollout_uid=request['rollout_uid']))
            rows={r['rollout_uid']:r for r in execute(requests,'return_repair_'+benchmark+'_'+method)}
            summaries=[]
            for category in sorted({str(o['category'])for o in owners}):
                record=dict(category=category)
                for label in ('before','after'):
                    planned=[o for o in owners if str(o['category'])==category and o['variant']==label];observed=[rows[o['rollout_uid']]for o in planned if o['rollout_uid']in rows]
                    record[label]=dict(planned=len(planned),observed=len(observed),success=sum(r['success']for r in observed),unhandled=sum(unhandled(r)for r in observed),uids=[o['rollout_uid']for o in planned])
                summaries.append(record)
            groups.append(dict(benchmark=benchmark,method=method,defaults=DEFAULTS[method],categories=summaries))
            output=dict(time=datetime.datetime.now().astimezone().isoformat(),complete=False,groups=groups,history_policy='Copied old3069paper labels lack matching A and nonrepresentative B training cases at current author-entry defaults. Fresh before/after requests use identical physical states/options, differing only adapter return handling; no repaired-label reuse.',source_release_sha256=sha(HERE/'EXECUTION_FROZEN.json'))
            dump(path,output);status('matched_training_solo_return_repair',len(groups),2*len(active_methods()),'训练场景状态0逐机器人单机；前后相同物理参数，只变返回值拦截。历史不兼容标签不混用。')
    output['complete']=all(v['observed']==v['planned']for g in groups for row in g['categories']for v in (row['before'],row['after']))
    dump(path,output);return output

if __name__=='__main__':run()
