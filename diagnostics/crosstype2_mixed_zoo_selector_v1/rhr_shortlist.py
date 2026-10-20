from common import *


def main():
    prereg();groups=[];sources=[]
    source=D/'blackbox_addendum_v1/RIGHT_HAND_SELECTED.json';sources.append(source)
    for name,value in read(source).items():
        if isinstance(value,dict) and 'rule' in value:groups.append(dict(source=str(source),group=name,rule=value['rule'],lower=value['tuning_lower']))
    for source in sorted((D/'blackbox_families_v1').glob('pilot_*_FAIR_RULES.json')):
        sources.append(source)
        for name,value in read(source).items():
            if isinstance(value,dict) and 'selected' in value:
                selected=value['selected'];groups.append(dict(source=str(source),group=name,rule=selected['rule'],lower=selected['lower']))
    fixed=dict(mode='progress',steps=40,speed=.35);bins={}
    for group in groups:
        if group['rule']==fixed:continue
        key=json.dumps(group['rule'],sort_keys=True)
        bins.setdefault(key,[]).append(group)
    ranked=[dict(rule=json.loads(key),selected_group_count=len(value),mean_tuning_lower=float(np.mean([v['lower'] for v in value])),groups=value) for key,value in bins.items()]
    mode_order={'always':0,'slow':1,'progress':2}
    ranked.sort(key=lambda v:(-v['selected_group_count'],-v['mean_tuning_lower'],v['rule']['speed'],v['rule']['steps'],mode_order[v['rule']['mode']]))
    write(HERE/'RHR_SHORTLIST.json',dict(utc=utc(),fixed=fixed,RHR_b=ranked[0]['rule'],RHR_c=ranked[1]['rule'],ranking=ranked,groups=groups,locked_data_used=False,source_sha256={str(s):sha(s) for s in sources}))
    seal('RHR_SHORTLIST.receipt.json',[HERE/'RHR_SHORTLIST.json',HERE/'rhr_shortlist.py'],one_time_history_only=True)
    print(json.dumps({k:read(HERE/'RHR_SHORTLIST.json')[k] for k in ('fixed','RHR_b','RHR_c')}))


if __name__=='__main__':main()
