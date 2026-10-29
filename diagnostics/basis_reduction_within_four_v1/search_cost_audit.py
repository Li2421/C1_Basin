"""Recompute search cost and completion times from existing Oracle records only."""
from pathlib import Path
from statistics import mean
import datetime
import json

HERE = Path(__file__).resolve().parent


def audit():
    codes = ('four_2', 'four_3', 'four_5')
    states = {c: json.loads((HERE / f'SEARCH_{c}_screen.json').read_text()) for c in codes}
    successes = {
        c: {tuple(g['identity']): g for g in s['groups'] if g['success']}
        for c, s in states.items()
    }
    common = set.intersection(*(set(gs) for gs in successes.values()))
    rows = []
    for code, state in states.items():
        groups = state['groups']
        assert len(groups) == 272
        assert all(g['success'] or g['searched'] == 64 for g in groups)
        good = list(successes[code].values())
        assert all(g['searched'] == g['first_success'] for g in good)
        total = sum(g['searched'] for g in groups)
        rows.append(dict(
            code=code, cases=len(groups), successes=len(good), failed_at_64=len(groups)-len(good),
            searched_total=total,
            success_first_index_mean=mean(g['first_success'] for g in good),
            all_case_searched_mean=total/len(groups),
            success_completion_seconds_mean=mean(g['solution']['result']['completion_time'] for g in good),
            common_success_cases=len(common),
            common_success_completion_seconds_mean=mean(successes[code][k]['solution']['result']['completion_time'] for k in common),
            all_case_capped_seconds_mean=mean(
                g['solution']['result']['completion_time'] if g['success']
                else g['queries'][-1]['result']['horizon_time'] for g in groups),
        ))
        assert total == sum(g['first_success'] for g in good) + 64*(len(groups)-len(good))
    out = dict(time=datetime.datetime.now().astimezone().isoformat(), rows=rows,
               common_success_identities=[list(k) for k in sorted(common)],
               interpretation='Direct ordered eta search, stop at first success; completion times are simulation seconds of that first success, not the fastest searched trajectory. All-case search cost includes all64queries for failures. No new rollouts.',
               statistical_limit='Descriptive matched times only; no significance claim or change to frozen winner selection.')
    (HERE/'SEARCH_COST_AUDIT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    return out


if __name__ == '__main__':
    print(json.dumps(audit()['rows'], ensure_ascii=False, indent=2))
