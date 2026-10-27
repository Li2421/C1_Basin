from common import *
from stages import read_result
from design import *
import numpy as np,subprocess,os

def collision_types(r,c):
    if 'collision_types'in r:return {k:int(r['collision_types'].get(k,0))for k in ('agent','static','moving')}
    if not r['collision']:return dict(agent=0,static=0,moving=0)
    # Read-only diagnostic from the stored first-collision step, using exactly
    # the frozen swept-clearance and endpoint-validity criteria. No env.step.
    from single_integrator.environment import point_segment_distance,segment_distance
    from flow_scene import LiteratureGeometry
    from legacy_states import LiteratureConfig
    with np.load(r['trace_path'],allow_pickle=False)as z:p=z['positions']
    before,after=p[-2:];cfg=LiteratureConfig(**c['scenario_payload']['config']);g=LiteratureGeometry(cfg)
    i,j=np.triu_indices(len(before),1)
    pair=float(point_segment_distance(np.zeros(2),before[i]-before[j],after[i]-after[j]).min()-2*cfg.agent_radius-cfg.agent_collision_margin)if len(i)else float('inf')
    wall=float(segment_distance(before[:,None],after[:,None],g.walls[None,:,0],g.walls[None,:,1]).min()-g.margin)if len(g.walls)else float('inf')
    flags=dict(agent=int(pair<=0),static=int(wall<=0 or not g.valid_points(after).all()),moving=0)
    assert any(flags.values()),('UNCLASSIFIED_STORED_COLLISION',c['rollout_uid'])
    return flags

def database():
 from scenes import make_new
 library={s['name']:s for s in read(HERE/'SCENE_LIBRARY.json')};stages=[s for s in ['capability','capability_mild','main']if(HERE/f'{s}_CASES.json').exists()];seen={};timings={};collisions={};tracehash={};counts={};metadata={}
 with dbm.connect(True)as db:
  for stage in stages:
   cases=read(HERE/f'{stage}_CASES.json');reg=read(HERE/f'{stage}_REGISTRATION.receipt.json');assert reg['original_rows_before']==reg['original_rows_after'];counts[stage]=len(cases)
   for c in cases:
    rid=c['rollout_uid'];name=c['state']['scene'];ctl=c['controller_payload']['nominal']['controller'];nominal=next(t for t in controllers()if t['controller']==ctl);action=next(a for a in menu()if a['key']==c['requested_action']);rebuilt=make_new(library[name],nominal,c['state'],action,c['seed']);assert canonical(rebuilt)==canonical(c)
    for table,key,col,payload in [('scenario','scenario_uid','metadata_json',c['scenario_payload']),('controller_config','controller_uid','config_json',c['controller_payload'])]:
     identity=(table,c[key])
     if identity not in metadata:metadata[identity]=db.execute(f'SELECT {col} FROM {table} WHERE {key}=?',(c[key],)).fetchone()[0]
     assert metadata[identity]==canonical(payload)
    state=db.execute('SELECT * FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone();physical=canonical({k:c['state'][k]for k in ['positions','velocities','goals']});assert state['physical_state_json']==physical and state['scenario_uid']==c['scenario_uid']
    if rid in seen:continue
    r=read_result(stage,c);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();assert row is not None
    assert canonical(r['case'])==canonical(c)
    for key in ['state_uid','controller_uid','eta_uid','seed_key']:assert row[key]==r[key]==c[key]
    for key in ['success','collision','timeout','numerical_failure','deadlock']:assert row[key]==r.get(key,0)
    assert row['outcome']==r['outcome'].lower();assert row['episode_length']==r['episode_length']or(r['episode_length']==0 and row['episode_length']is None)
    assert row['raw_record_hash']==hashlib.sha256(dbm.canonical(r).encode()).hexdigest();assert not row['conflict_quarantined']and not r.get('deadlock',0)
    if r['success']:assert not r['collision']and not r['numerical_failure']
    eta=db.execute('SELECT * FROM eta WHERE eta_uid=?',(c['eta_uid'],)).fetchone();assert eta['canonical_hex']==dbm.eta_identity(c['eta_a'])[2]
    assert sha(r['trace_path'])==r['trace_sha256'];tracehash[r['trace_path']]=r['trace_sha256']
    with np.load(r['trace_path'],allow_pickle=False)as z:
     assert np.array_equal(z['positions'][0],c['state']['positions'])and np.array_equal(z['goals'],c['state']['goals']);assert len(z['actions'])==r['episode_length']
    seen[rid]=dict(stage=stage,unknown=int(r['numerical_failure']),outcome=r['outcome']);timings[rid]=dict(scene=name,controller=ctl,seconds=float(r['elapsed_seconds']));collisions[rid]=collision_types(r,c)
 pending=list((cache_writer.ROOT/'journals'/EXP).glob('*.jsonl'))+list((cache_writer.ROOT/'journals'/EXP).glob('*.tmp'));assert not pending
 new={c['rollout_uid']for s in stages for c in read(HERE/f'{s}_MISSING_CASES.json')};elapsed=np.array([t['seconds']for rid,t in timings.items()if rid in new]);mainnew=len(read(HERE/'main_MISSING_CASES.json'));assert mainnew<=32000
 write(HERE/'DB_POSTFLIGHT.json',dict(utc=utc(),passed=True,stage_requests=counts,unique_requests=len(seen),new_requests=len(new),new_main_requests=mainnew,all_requests_present=True,all_cases_exactly_reconstructed=True,all_metadata_and_identities_equal=True,all_raw_database_record_hashes_equal=True,all_trace_hashes_equal=True,no_unmerged_experiment_journals=True,preexisting_metadata_unchanged_at_registration=True,new_unknowns=sum(r['unknown']for r in seen.values())))
 write(HERE/'COLLISION_CLASSIFICATION.json',dict(records=collisions,counts={k:sum(v[k]for v in collisions.values())for k in ['agent','static','moving']}));write(HERE/'VERIFIED_RESULT_PROVENANCE.json',seen);write(HERE/'VERIFIED_TRACE_SHA256.json',tracehash)
 write(HERE/'ACTUAL_COST.json',dict(new_requests=len(new),new_main_requests=mainnew,episode_wall_cpu_hours=float(elapsed.sum()/3600),maximum_episode_seconds=float(elapsed.max()),quantiles={str(q):float(np.quantile(elapsed,q))for q in [.5,.9,.95,.99]},per_request=timings,scope='Episode elapsed time only; excludes import,compression,queueing,training.'))

def historical_database():
 receipt=read(HERE/'HISTORICAL_ROW_PROVENANCE.json');records=sorted(receipt['records'],key=lambda r:r['rollout_uid']);h=hashlib.sha256();count=0;select=receipt['selection_sql'].split('WHERE',1)[0]
 with dbm.connect(True)as db:
  for original in read(HERE/'REGISTERED_TRAINING_STATES.receipt.json')['states']:
   current=db.execute('SELECT state_uid,scenario_uid,physical_state_json FROM state WHERE state_uid=?',(original['state_uid'],)).fetchone();assert current is not None and canonical(dict(current))==canonical(original)
  for g in read(HERE/'OLD_TRAINING_POOL.json'):
   for uid_ in g['scenario_uids']:
    sc=db.execute('SELECT name,metadata_json FROM scenario WHERE scenario_uid=?',(uid_,)).fetchone();assert sc['name']==g['source_version']and json.loads(sc['metadata_json'])['config']==g['config']
  for begin in range(0,len(records),500):
   batch=records[begin:begin+500];expected={r['rollout_uid']:r['sha256']for r in batch};sql=select+'WHERE r.rollout_uid IN ('+','.join('?'for _ in batch)+') ORDER BY r.rollout_uid';seen=set()
   for row in db.execute(sql,list(expected)):
    r=dict(row);digest=hashlib.sha256(canonical(r).encode()).hexdigest();assert digest==expected[r['rollout_uid']];h.update((r['rollout_uid']+digest).encode());count+=1;seen.add(r['rollout_uid'])
   assert seen==set(expected)
 assert count==receipt['record_count']and h.hexdigest()==receipt['sha256']
 write(HERE/'HISTORICAL_ROWS_END.receipt.json',dict(utc=utc(),passed=True,records=count,sha256=h.hexdigest(),only_whitelisted_training_versions=True,indexed_snapshot_row_audit=True,parallel_appends_do_not_change_frozen_snapshot=True))

def metadata_tree(root):
 result={}
 for base,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   p=Path(base)/name
   try:s=p.stat(follow_symlinks=False)
   except FileNotFoundError:continue
   result[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode)
 return result

def integrity():
 start=read(HERE/'INTEGRITY_START.json');protected=ROOT/'diagnostics/cross_scene_prospective_v1';expected={}
 def add(p,h):
  p=Path(p).resolve();assert not p.is_relative_to(protected)
  prior=expected.setdefault(str(p),h);assert prior==h,('SOURCE_CHANGED_BETWEEN_READS',str(p))
 for p,h in read(HERE/'REUSED_EVIDENCE_SHA256_START.json')['files'].items():add(p,h)
 for line in(HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():
  r=json.loads(line);add(r['path'],r['sha256'])
 for r in read(HERE/'TOOLING_COPY.receipt.json')['files']:add(r['source'],r['source_sha256']);assert sha(r['target'])==r['target_sha256']
 for r in read(HERE/'BACKEND_BUILD.json')['sources']:add(r['source'],r['source_sha256']);assert sha(r['target'])==r['target_sha256']
 assert sha(HERE/'native_resume/cbs_kernel_resume.so')==read(HERE/'BACKEND_BUILD.json')['native_sha256'];changed={p:dict(before=h,after=sha(p))for p,h in expected.items()if sha(p)!=h};frozen={}
 for n in ['EXECUTION_FROZEN.receipt.json','TRAINING_FROZEN.receipt.json','FROZEN.receipt.json']:
  frozen.update(read(HERE/n)['files'])
 frozen_changed={p:dict(before=h,after=sha(p))for p,h in frozen.items()if sha(p)!=h};prospective=metadata_tree(protected);prospective_changes={p:dict(before=start['prospective_metadata'].get(p),after=prospective.get(p))for p in set(start['prospective_metadata'])|set(prospective)if start['prospective_metadata'].get(p)!=prospective.get(p)}
 parallel=read(HERE/'PARALLEL_METADATA_START.json')['directories'];parallel_changes={}
 for directory,old in parallel.items():
  now=metadata_tree(directory);changes={p:dict(before=old.get(p),after=now.get(p))for p in set(old)|set(now)if old.get(p)!=now.get(p)}
  parallel_changes[directory]=changes
 write(HERE/'PARALLEL_METADATA_END.json',dict(utc=utc(),changes=parallel_changes,stat_only=True,changes_explicitly_excepted=True))
 for module in list(sys.modules.values()):
  filename=getattr(module,'__file__',None)
  if filename:assert not Path(filename).resolve().is_relative_to(protected)
 def normalize(status,repo):
  keep=[];skip=[]
  for line in status.splitlines():
   p=line[3:];absolute=str(Path(repo)/p);exception=(p=='docs/exact_engineering_acceleration_20261026.md'and line.startswith('??')and read(HERE/'EXTERNAL_DOCUMENT_EXCEPTION.receipt.json')['user_approved'])or absolute.startswith(str(HERE)+'/')or any(absolute.startswith(d.rstrip('/')+'/')for d in start['parallel_directories'])or(line.startswith('??')and p.startswith('docs/prompt_')and p.endswith('.md'))
   (skip if exception else keep).append(line)
  return keep,skip
 repos={};exceptions={}
 for repo,old in start['repositories'].items():
  status=subprocess.check_output(['git','-C',repo,'status','--porcelain']).decode();diff=subprocess.check_output(['git','-C',repo,'diff']);cached=subprocess.check_output(['git','-C',repo,'diff','--cached']);before,bs=normalize(old['status'],repo);after,es=normalize(status,repo)
  repos[repo]=dict(status=status,normalized_status_equal=before==after,diff_sha256=hashlib.sha256(diff).hexdigest(),cached_diff_sha256=hashlib.sha256(cached).hexdigest(),diff_equal=hashlib.sha256(diff).hexdigest()==old['diff_sha256'],cached_diff_equal=hashlib.sha256(cached).hexdigest()==old['cached_diff_sha256']);exceptions[repo]=dict(before=bs,after=es)
 allowed_prospective=not prospective_changes or str(protected)in start['parallel_directories'];passed=not changed and not frozen_changed and allowed_prospective and all(v['normalized_status_equal']and v['diff_equal']and v['cached_diff_equal']for v in repos.values())
 write(HERE/'INTEGRITY_END.json',dict(utc=utc(),repositories=repos,exceptions=exceptions,passed=passed));write(HERE/'INTEGRITY_COMPARE.json',dict(passed=passed,changed_inputs=changed,frozen_changed=frozen_changed,source_files=len(expected),frozen_files=len(frozen),reserved_prospective_only_stat=True,prospective_metadata_changes=prospective_changes,parallel_changes_excepted=True));write(HERE/'REUSED_EVIDENCE_SHA256_END.json',dict(files={p:sha(p)for p in expected}))
 if not passed:write(HERE/'STOP.receipt.json',dict(utc=utc(),reason='SOURCE_OR_FROZEN_INTEGRITY_FAILURE',changed_inputs=changed,frozen_changed=frozen_changed));raise RuntimeError('SOURCE_OR_FROZEN_INTEGRITY_FAILURE')
