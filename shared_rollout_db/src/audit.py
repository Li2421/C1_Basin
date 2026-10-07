from __future__ import annotations
import csv,json,sqlite3
from collections import defaultdict
from pathlib import Path
from .rollout_db import ROOT,connect
from .planner import preflight

def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True); fields=fields or (list(rows[0]) if rows else ['empty'])
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def make_dry_manifest(con,experiment_name,label):
    rows=con.execute('''SELECT DISTINCT r.state_uid,r.eta_uid,r.controller_uid,r.seed_key
      FROM rollout r JOIN rollout_source rs USING(rollout_uid) JOIN source_file sf USING(source_uid)
      JOIN experiment ex ON sf.experiment_uid=ex.experiment_uid
      WHERE ex.name=? AND r.conflict_quarantined=0 ORDER BY r.state_uid,r.eta_uid,r.controller_uid,r.seed_key''',(experiment_name,)).fetchall()
    groups={}
    for r in rows:groups.setdefault((r['state_uid'],r['eta_uid'],r['controller_uid']),[]).append(r['seed_key'])
    req=[{'state_uid':k[0],'eta_uid':k[1],'controller_uid':k[2],'seed_keys':v} for k,v in groups.items()]
    p=ROOT/'audits'/f'dry_run_{label}_manifest.json';dump(p,{'source_experiment':experiment_name,'requests':req});return p,len(rows)

def main():
  for d in ('audits','exports','journals'): (ROOT/d).mkdir(exist_ok=True)
  with connect() as con:
    counts={t:con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('scenario','state','eta','controller_config','rollout','aggregate_evidence','experiment','source_file','conflict')}
    quality={r[0]:r[1] for r in con.execute('SELECT compatibility_quality,COUNT(*) FROM rollout GROUP BY 1')}
    scenarios=[]
    for r in con.execute('''SELECT sc.scenario_uid,sc.name,COUNT(DISTINCT s.state_uid) unique_states,COUNT(DISTINCT r.eta_uid) unique_eta,
      COUNT(r.rollout_uid) seed_rollouts,SUM(CASE WHEN r.compatibility_quality='EXACT_REUSE' AND r.conflict_quarantined=0 THEN 1 ELSE 0 END) exact_reusable
      FROM scenario sc LEFT JOIN state s USING(scenario_uid) LEFT JOIN rollout r USING(state_uid) GROUP BY sc.scenario_uid,sc.name ORDER BY seed_rollouts DESC'''):scenarios.append(dict(r))
    write_csv(ROOT/'exports'/'scenario_summary.csv',scenarios)
    coverage=[dict(r) for r in con.execute('''SELECT sc.name scenario,r.state_uid,r.eta_uid,r.controller_uid,COUNT(*) n_trials,SUM(r.success) n_success,
      SUM(r.deadlock) n_deadlock,SUM(r.timeout) n_timeout,SUM(r.collision) n_collision,
      CASE WHEN COUNT(*)>=16 AND SUM(r.success)>=15 THEN 1 ELSE 0 END b15_observed
      FROM rollout r JOIN state s USING(state_uid) JOIN scenario sc USING(scenario_uid)
      WHERE r.compatibility_quality='EXACT_REUSE' AND r.conflict_quarantined=0 AND r.numerical_failure=0
      GROUP BY sc.name,r.state_uid,r.eta_uid,r.controller_uid''')]
    write_csv(ROOT/'exports'/'state_eta_coverage.csv',coverage)
    covsum=[]
    for name in ('ToyGiveWay','DoubleBottleneck_4A'):
      z=[r for r in coverage if r['scenario']==name];ready=[r for r in z if int(r['n_trials'])>=16];b15=[r for r in ready if int(r['n_success'])>=15]
      covsum.append({'scenario':name,'state_eta_pairs':len(z),'pairs_ge16_compatible_seeds':len(ready),'reuse_coverage_fraction':len(ready)/len(z) if z else 0,'observed_B15_pairs':len(b15)})
    write_csv(ROOT/'audits'/'historical_coverage.csv',covsum)
    conflicts=[dict(r) for r in con.execute('SELECT conflict_uid,entity_type,identity_key,source_file,status,created_at FROM conflict ORDER BY source_file')]
    write_csv(ROOT/'audits'/'conflicts.csv',conflicts)
    incompatible=[dict(r) for r in con.execute('''SELECT r.rollout_uid,sc.name scenario,r.state_uid,r.eta_uid,r.controller_uid,r.compatibility_quality,r.original_source_file
      FROM rollout r JOIN state s USING(state_uid) JOIN scenario sc USING(scenario_uid) WHERE r.compatibility_quality!='EXACT_REUSE' OR r.conflict_quarantined=1''')]
    write_csv(ROOT/'audits'/'incompatible_records.csv',incompatible)
    strong_agg=con.execute('SELECT COUNT(*) FROM aggregate_evidence WHERE certified_b15=1 AND conflict_quarantined=0').fetchone()[0]
    strong_q64=con.execute("SELECT COUNT(*) FROM aggregate_evidence WHERE n_trials=64 AND n_success>=63 AND conflict_quarantined=0").fetchone()[0]
    seed_q64=con.execute('''SELECT COUNT(*) FROM (SELECT state_uid,eta_uid,controller_uid,COUNT(*) n,SUM(success) s FROM rollout
      WHERE compatibility_quality='EXACT_REUSE' AND conflict_quarantined=0 AND numerical_failure=0 GROUP BY 1,2,3 HAVING n>=64 AND s>=n-1)''').fetchone()[0]
    # Dry runs use exact requests reconstructed from immutable original experiment artifacts.
    specs=[('orthoflow3_shared_eta_codebook_v1','toy'),('orthoflow3_db_shared_mode_transfer_v1','db'),('orthoflow3_toy_db_conditional_generator_v1','recent')]
    dry=[]
    for exp,label in specs:
      p,n=make_dry_manifest(con,exp,label);res=preflight(p);dump(ROOT/'audits'/f'dry_run_{label}_result.json',res)
      dry.append({'label':label,'experiment':exp,'requested':n,**res['summary']})
    write_csv(ROOT/'audits'/'dry_run_summary.csv',dry)
    prior_summary=ROOT/'audits'/'audit_summary.json'
    stats=(json.load(open(prior_summary))['ingest'] if prior_summary.exists() else json.load(open(ROOT/'audits'/'ingest_stats.json')))
    report={'counts':counts,'ingest':stats,'quality':quality,'scenario_coverage':covsum,'certified_b15_aggregate':strong_agg,
            'historical_q64_strong_aggregate':strong_q64,'seed_level_q64_strong_pairs':seed_q64,'dry_runs':dry}
    dump(ROOT/'audits'/'audit_summary.json',report)
    md=['# Global rollout database migration audit','',f"Scanned {stats['experiment_dirs']} experiment directories and {stats['files_scanned']} candidate files.",
        f"Imported {counts['rollout']:,} unique seed-level rollouts and {counts['aggregate_evidence']:,} unique aggregate records; {stats['duplicates']:,} duplicate observations were deduplicated.",
        f"Conflicts quarantined: {counts['conflict']:,}. Ambiguous/incompatible records remain indexed but are excluded from automatic reuse.",'','## Scenario coverage','',
        '| scenario | state-eta pairs | >=16 compatible seeds | coverage | observed B15 |','|---|---:|---:|---:|---:|']
    for r in covsum:md.append(f"| {r['scenario']} | {r['state_eta_pairs']} | {r['pairs_ge16_compatible_seeds']} | {r['reuse_coverage_fraction']:.3f} | {r['observed_B15_pairs']} |")
    md += ['','## Dry-run validation','', '| case | requested | exact reused | missing | ambiguous |','|---|---:|---:|---:|---:|']
    for r in dry:md.append(f"| {r['label']} | {r['requested']} | {r['exact_reusable']} | {r['genuinely_missing']} | {r['ambiguous']} |")
    md += ['','No rollout was executed. Each dry run reconstructed exact historical request keys, then queried the global index. Semantic mismatches remain non-reusable.']
    (ROOT/'audits'/'migration_report.md').write_text('\n'.join(md)+'\n')
    (ROOT/'audits'/'dry_run_validation.md').write_text('\n'.join(md[md.index('## Dry-run validation'):])+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
