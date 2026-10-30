"""Find archived candidate evidence without deciding reuse by its outcome."""
import guard
from guard import HERE
import collections
import hashlib
import json

BASE=HERE/'inputs/diagnostics'


def read(path):return json.loads(path.read_text())


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def signature(scene,controller,index,spec):
    return hashlib.sha256(canonical([scene,controller,index,spec]).encode()).hexdigest()


def pools():
    main=read(BASE/'bottleneck_one_way_admission_v1/FAMILIES_THIS_ROUND.json')['families'][0]
    four=next(f for f in read(BASE/'scene_types_oracle_coverage_v3/FAMILIES_64.json')['families'] if f['index']==4)
    assert four['columns']==['G','R','T1','C2']
    return dict(main=main['candidate_pool'][:32],four=four['candidate_pool'][:32])


def main():
    candidates=pools()
    representative={x['scene'] for x in read(BASE/'scene_types_oracle_coverage_v3/REPRESENTATIVE17_FROZEN.json')['selected']}
    train={e['scene']['name'] for e in read(BASE/'scene_types_oracle_coverage_v3/SCENES.json')['entries'] if e['scene']['name']not in representative}
    wanted=collections.defaultdict(list)
    for family,values in candidates.items():
        for i,spec in enumerate(values,1):wanted[canonical(spec)].append((family,i))
    rows={};index=collections.defaultdict(list);counts=collections.Counter();duplicate=0
    for study in ('scene_types_oracle_coverage_v3','bottleneck_one_way_admission_v1'):
        stream=BASE/study/'candidates.jsonl'
        with stream.open()as f:
            for line_no,line in enumerate(f,1):
                r=json.loads(line)
                if r.get('solo')is not None or r.get('seed',0)!=0 or r.get('controller')not in ('centralized_sampling','gradient_mpc'):continue
                if r.get('scene')not in train or not 0<=r.get('state_index',-1)<8:continue
                spec=r.get('spec');key=canonical(spec)
                if key not in wanted:continue
                uid=r['rollout_uid']
                if uid in rows:
                    previous=rows[uid]['result']
                    assert all(previous.get(k)==r.get(k)for k in ('success','collision','timeout','numerical_failure','episode_length','trace_sha256'))
                    duplicate+=1
                    continue
                sig=signature(r['scene'],r['controller'],r['state_index'],spec)
                rows[uid]=dict(source_stream=str(stream.relative_to(HERE)),source_line=line_no,
                               source_rollout_uid=uid,signature=sig,candidate_origins=wanted[key],result=r)
                index[sig].append(uid)
                for family,i in wanted[key]:counts[family]+=1
    out=dict(records=rows,signature_index=dict(index),counts=dict(counts),unique_source_rollouts=len(rows),duplicates_omitted=duplicate,
             candidate_pool_limit=32,training_scenes=len(train),new_rollouts=0,
             certified_reuse=False,
             required_next='Verify physical/config/controller/RNG/source fingerprints, sample50-step prefixes, then official target preflight. Multiple versions remain separate. No choice based on success.')
    (HERE/'OURS_HISTORY_INDEX.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('Archived C1 candidates:',len(rows),'unique source UIDs;',len(index),'physical/controller/spec keys. Compatibility not certified.')


if __name__=='__main__':main()
