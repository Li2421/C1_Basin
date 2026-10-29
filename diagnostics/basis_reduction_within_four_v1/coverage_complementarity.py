"""Per-case complementarity of completed 64-candidate families; analysis only."""
import guard
from guard import HERE
import collections
import datetime
import json


def audit():
    read = lambda name: json.loads((HERE / name).read_text())
    combinations = {c['code']: c for c in read('COMBINATIONS.json')['combinations']}
    scenes = {(e['type_id'], e['level']): e['scene'] for e in read('SCENES.json')['entries']}
    complete = {}
    for code in combinations:
        path = HERE / f'SEARCH_{code}_screen.json'
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state['complete64'] and len(state['groups']) == 272:
            complete[code] = {tuple(g['identity']): g for g in state['groups']}
    assert complete
    coverage = {code: sum(g['success'] for g in groups.values()) for code, groups in complete.items()}
    maximum = max(coverage.values())
    ties = [code for code in complete if coverage[code] == maximum]
    leader = min(ties, key=lambda code: (combinations[code]['dimension'], code))
    identities = list(complete[leader])
    assert all(set(groups) == set(identities) for groups in complete.values())
    records = []
    outcomes = collections.Counter()
    for key in identities:
        group = complete[leader][key]
        if group['success']:
            continue
        assert group['searched'] == 64
        outcomes.update(q['result']['outcome'] for q in group['queries'])
        rescued = []
        for code, groups in complete.items():
            alternative = groups[key]
            if alternative['success']:
                query = alternative['queries'][alternative['first_success'] - 1]
                rescued.append(dict(code=code, name=combinations[code]['name'], first_success=alternative['first_success'],
                                    rollout_uid=query['result']['rollout_uid'],
                                    spec=combinations[code]['candidate_pool'][alternative['first_success'] - 1]))
        records.append(dict(identity=list(key), title=scenes[key[:2]]['title'],
                            rescued_by_completed_families=rescued, any_rescue=bool(rescued)))
    union = sum(any(groups[key]['success'] for groups in complete.values()) for key in identities)
    by_type = []
    for t in sorted({r['identity'][0] for r in records}):
        selected = [r for r in records if r['identity'][0] == t]
        by_type.append(dict(type_id=t, title=selected[0]['title'], uncovered=len(selected),
                            rescued=sum(r['any_rescue'] for r in selected), still_uncovered=sum(not r['any_rescue'] for r in selected)))
    output = dict(time=datetime.datetime.now().astimezone().isoformat(), leader=dict(code=leader, name=combinations[leader]['name'],
                  dimension=combinations[leader]['dimension'], coverage=maximum), tied_highest=[dict(code=c, name=combinations[c]['name']) for c in ties],
                  completed_family_coverage=coverage, cases=272, diagnostics_only_union=union,
                  leader_failure_query_outcomes=dict(outcomes), by_type=by_type, records=records,
                  still_uncovered=[r['identity'] for r in records if not r['any_rescue']],
                  scope='Only families with completed64 searches on all272 cases; no per-case method selection is reported as a fixed-family score.',
                  new_rollouts=0)
    (HERE / 'COVERAGE_COMPLEMENTARITY.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    return output


if __name__ == '__main__':
    result = audit()
    print(json.dumps({k: result[k] for k in ('leader', 'diagnostics_only_union', 'by_type')}, ensure_ascii=False))
