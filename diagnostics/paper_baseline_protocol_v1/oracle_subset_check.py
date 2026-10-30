"""Finite-prefix consistency against matched frozen fixed-policy successes."""
import guard
from guard import HERE
import json


def check(fixed_rows,methods):
    checks=[];violations=[]
    for method in methods:
        name=method['method'];groups={}
        for group in method['groups']:
            nominal=group['nominal']['type_code']if group['nominal']is not None else'paper'
            key=(group['scene'],group['state'],nominal);assert key not in groups
            groups[key]=group
        rows=[r for r in fixed_rows if r['method']==name];inside=0;successes=0
        for row in rows:
            if row['menu_index']>16:continue
            inside+=1
            result=row.get('result')
            if not method.get('complete')or not result or not result['success']:continue
            successes+=1;key=(row['scene'],row['state'],row['nominal']);group=groups.get(key)
            if group is None or not group['success']or group['first_success']>row['menu_index']:
                violations.append(dict(method=name,key=key,fixed_index=row['menu_index'],reason='Complete prefix failed to contain a matched fixed-policy success.'));continue
            for query in group['queries']:
                if query['index']==row['menu_index']and query['query_uid']!=row['query_uid']:
                    violations.append(dict(method=name,key=key,fixed_index=row['menu_index'],reason='Identical frozen candidate has a different execution UID.'))
        checks.append(dict(method=name,oracle_complete=method.get('complete'),fixed_logical_rows=len(rows),
            fixed_rows_inside_prefix=inside,inside_prefix_fixed_successes_verified=successes,
            fixed_rows_outside_prefix=len(rows)-inside))
    return dict(passed=not violations,methods=checks,violations=violations,
        interpretation='When a globally selected setting is inside the same frozen16-point pool and the Oracle run is complete, every matched fixed-policy success must be covered no later than that index. Tuned17–32settings are explicitly outside this guarantee; incomplete runs do not imply failure.')


def run():
    fixed=json.loads((HERE/'test_tables/test_B_global_tuned.json').read_text())['rows']
    oracle=json.loads((HERE/'ORACLE16_RESULTS.json').read_text())['methods'];result=check(fixed,oracle)
    (HERE/'ORACLE_SUBSET_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Oracle/fixed-policy subset inconsistency; preserve raw data and diagnose ORACLE_SUBSET_CHECK.json')
    return result
