"""Read-only, completion-censored cost snapshot; never launches science work."""
import collections
import datetime
import json
import os
from pathlib import Path

H = Path(__file__).resolve().parent


def load(p):
    return json.loads(p.read_text())


def main():
    guard = load(H / 'LOCAL_CPU_GUARD_STATUS.json')
    if isinstance(guard.get('local_cpu'), int):
        os.sched_setaffinity(0, {guard['local_cpu']})
    now = datetime.datetime.now(datetime.timezone.utc)
    out = dict(utc=now.isoformat(), scope='Read-only optional monitoring; landed samples are completion-censored, not final success estimates', stages={})
    for folder in sorted(H.glob('optional_*_lit_*')):
        if not folder.is_dir() or not (folder / 'FINE_GROUP_INDEX.json').exists():
            continue
        index = load(folder / 'FINE_GROUP_INDEX.json')
        raw = {p.stem: p for p in (folder / 'raw').glob('*.json')}
        journal = {p.stem for p in (folder / 'journal_receipts').glob('*.json')}
        landed = raw.keys() & journal
        types = collections.defaultdict(lambda: dict(planned=0, landed=0, candidates=collections.defaultdict(list), active=[], by_N=collections.defaultdict(lambda: dict(planned=0, landed=0))))
        files = {}
        for e in index:
            if e['file'] not in files:
                files[e['file']] = load(Path(e['file']))
            cases = files[e['file']][e['start']:e['end']]
            kind = cases[0]['controller_payload']['nominal']['type_code']
            row = types[kind]
            row['planned'] += len(cases)
            row['landed'] += sum(c['rollout_uid'] in landed for c in cases)
            for c in cases:
                cohort = row['by_N'][c['state']['N']]
                cohort['planned'] += 1
                cohort['landed'] += int(c['rollout_uid'] in landed)
            # Preserve each two-seed claim and stratify by controller/method/N.
            # Periodic queue sampling aliases the seven-N scene ordering.
            if all(c['rollout_uid'] in landed for c in cases):
                key = (cases[0]['state']['N'], cases[0]['controller_payload']['nominal']['controller'], cases[0]['method'])
                row['candidates'][key].append([c['rollout_uid'] for c in cases])
            claim = folder / 'fine_claims' / e['group']
            if claim.exists() and not (claim / 'COMPLETE.json').exists() and (claim / 'OWNER.json').exists():
                owner = load(claim / 'OWNER.json')
                missing = [c for c in cases if c['rollout_uid'] not in landed]
                if missing:
                    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(owner['utc'])).total_seconds()
                    c = missing[0]
                    row['active'].append(dict(job=owner['job'], pid=owner['pid'], claim_age_seconds=round(age, 1), controller=c['controller_payload']['nominal']['controller'], method=c['method'], N=c['state']['N'], state=c['state']['index']))
        for row in types.values():
            candidates = row.pop('candidates')
            sample = []
            for groups in candidates.values():
                for gi in sorted({0, len(groups)-1}):
                    for rid in groups[gi]:
                        r = load(raw[rid])
                        sample.append((r['elapsed_seconds'], r['episode_length'], r['outcome'], bool(r.get('deterministic_source')), r['case']['state']['N']))
            durations = [r[0] for r in sample]
            row.update(remaining=row['planned']-row['landed'], cost_samples=len(sample), sample_physical=sum(not r[3] for r in sample), sampled_mean_seconds_per_logical_request=sum(durations)/len(durations) if durations else None, sampled_max_seconds=max(durations, default=None), sampled_outcomes=dict(collections.Counter(r[2] for r in sample)))
            for n, cohort in row['by_N'].items():
                ds = [r[0] for r in sample if r[4] == n]
                cohort.update(remaining=cohort['planned']-cohort['landed'], cost_samples=len(ds), sampled_mean_seconds=sum(ds)/len(ds) if ds else None, sampled_max_seconds=max(ds, default=None))
        out['stages'][folder.name] = dict(types=types, landed=len(landed), raw_without_journal=len(raw.keys()-journal))
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
