"""Conditional complete-cohort timing supplement; immutable old labels win."""
from core import *
from dispatch import execute,DeadlineReached
from resource_watch import available
from scene_catalog import catalog,frozen_eta
from learned_setup import rows as learned_rows
from request_factory import case
from consistent_ours import engine
import time,datetime


def run():
    assert(HERE/'FORMAL_TEST_STARTED.json').exists()
    receipt=HERE/'TIME_BACKFILL_DECISION.json'
    if receipt.exists() and read(receipt).get('finished'):return read(receipt)
    started=time.time();deadline=started+3540
    scenes=catalog('A',version='original');models={r['controller']:r for r in learned_rows()}
    labels=read(HERE/'FROZEN_HISTORICAL_LABELS.json')['records']+read(HERE/'FROZEN_VALIDATION_LABELS.json')['records']
    successes=[r for r in labels if r['success']]
    assert len(successes)==7145
    costs={}
    # Already-required original-physics episodes, not new performance pilots.
    for filename,method in [('ENV_CHECK.json','只加安全投影'),('REUSE_CHECK.json','学习选修正参数')]:
        samples=[]
        for row in read(HERE/filename)['records']:
            path=HERE/'raw'/(row['target_query_uid']+'.json')
            if not path.exists():continue
            result=read(path)
            with np.load(result['trace'])as trace:
                values=trace['step_seconds']
            if len(values):samples.extend(values.tolist())
        if samples:costs[method]=float(np.quantile(samples,.95))*1.5
    if '只加安全投影'in costs:costs['经典目标右侧规则']=1.75*costs['只加安全投影']
    cpus=available()
    prediction=sum((scenes[r['scene']]['config']['max_steps']*costs.get(r['method'],np.inf)+2.)for r in successes)/max(cpus,1)+120.
    remaining=50000-read(HERE/'BUDGET.json')['actual_attempts']
    decision=dict(time=datetime.datetime.now().astimezone().isoformat(),success_rows=len(successes),all_historical_rows=len(labels),
        cpu_workers_available=cpus,conservative_full_incremental_seconds=prediction if np.isfinite(prediction)else None,
        measured_step_envelopes=costs,forecast='95th percentile already-required per-step wall time,50%headroom,full scene horizon,2sstartup/episode; RHRuses1.75xsafety envelope. Estimate, not a runtime certificate.',
        original_labels_authoritative=True,maximum_seconds=3600,deadline_seconds=3540)
    if cpus<=0 or not np.isfinite(prediction)or prediction>=3600 or remaining<len(successes):
        decision.update(executed=False,finished=True,reason='Full supplement not conservatively below1hour or insufficient remaining rollout budget; times stay missing.')
        dump(receipt,decision);return decision
    requests=[];metadata=[]
    for old in successes:
        eta=frozen_eta(old['scene'])[1]if old['method']=='学习选修正参数'else[0.,0.,0.]
        request=case('A',scenes[old['scene']],models[old['name']],old['state'],spec=dict(eta=eta,coefficients={}),phase='timing_backfill',purpose='authorized_old_success_time_only')
        request=engine(request,old['method']);requests.append(request)
        metadata.append(dict(identity=[old['scene'],old['state'],old['name'],old['method']],query_uid=request['rollout_uid'],old_outcome=old['outcome']))
    dump(receipt,dict(decision,executed=True,finished=False))
    stopped=False
    try:rows=execute(requests,'conditional_old_success_time_backfill',deadline=deadline)
    except DeadlineReached:
        stopped=True;rows=[]
    found={r.get('target_query_uid',r['rollout_uid']):r for r in rows}
    # A stopped dispatch retains every completed raw row without relabeling the
    # missing remainder as failure or shrinking the requested dataset.
    for item in metadata:
        path=HERE/'raw'/(item['query_uid']+'.json')
        if item['query_uid']not in found and path.exists():found[item['query_uid']]=read(path)
    records=[dict(item,result=found.get(item['query_uid']))for item in metadata]
    mismatches=[r['identity']for r in records if r['result']and(not r['result']['success']or r['result']['outcome']!=r['old_outcome'])]
    dump(HERE/'TIME_BACKFILL_RESULTS.json',dict(complete=all(r['result']for r in records)and not mismatches,
        records=records,original_labels_unchanged=True,mismatches=mismatches,deadline_stopped=stopped,elapsed_wall_seconds=time.time()-started))
    decision.update(executed=True,finished=True,complete=all(r['result']for r in records)and not mismatches,
                    elapsed_wall_seconds=time.time()-started,stopped_at_deadline=stopped)
    dump(receipt,decision)
    return decision
