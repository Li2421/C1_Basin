"""Official database/raw/trace audit and protected-source final integrity."""
from common import *
from stages import read_result
import numpy as np,subprocess

@lru_cache(None)
def case_index(stage):return{c['rollout_uid']:c for c in read(HERE/f'{stage}_CASES.json')}
def result(stage,rid):return read_result(stage,case_index(stage)[rid])

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
    from design import CONTROL_IDS,controllers,menu,OLD,NEW_NAMES
    scene_index={s['name']:s for s in read(HERE/'SCENES.json')}
    seen={};counts={};hashes={};collision={};checkpoint_files=set();metadata={};timings={};missing_requests=[]
    executed={c['rollout_uid']for stage in('capability','main','proposal')for c in read(HERE/f'{stage}_MISSING_CASES.json')}
    def digest(p):
        p=str(p)
        if p not in hashes:hashes[p]=sha(p)
        return hashes[p]
    with dbm.connect(True)as db:
        for stage in ('capability','main','proposal'):
            cases=read(HERE/f'{stage}_CASES.json');counts[stage]=len(cases)
            reg=read(HERE/f'{stage}_REGISTRATION.receipt.json');assert reg['original_rows_before']==reg['original_rows_after']
            for c in cases:
                rid=c['rollout_uid'];name=c['state']['scene'];ctl=c['controller_payload']['nominal']['controller']
                scene=scene_index[name]
                nominal=next(t for t in controllers()if t['controller']==ctl);action=next((a for a in menu()if a['key']==c['requested_action']),dict(key=c['requested_action'],eta_a=c['eta_a']))
                assert ctl in CONTROL_IDS
                if stage=='capability'and name not in NEW_NAMES:
                    from stages import old_cap_cases
                    original=next(x for x in old_cap_cases()if x['rollout_uid']==rid);assert canonical(c)==canonical(original)
                elif name.startswith('lit_'):
                    from requests_corrected import request
                    assert canonical(request(name,ctl,c['state']['index'],c['eta_a']))==canonical(c)
                else:
                    assert canonical(nominal)==canonical(c['controller_payload']['nominal'])
                    rebuilt=make_new(scene,nominal,c['state'],action,c['seed']);assert canonical(rebuilt)==canonical(c),('PLANNED_CONSTRUCTOR_DIFF',rid)
                for table,key,expect,col in [('scenario','scenario_uid',canonical(c['scenario_payload']),'metadata_json'),
                    ('controller_config','controller_uid',canonical(c['controller_payload']),'config_json')]:
                    identity=(table,c[key]);prior=metadata.get(identity)
                    if prior is None:
                        row=db.execute(f'SELECT * FROM {table} WHERE {key}=?',(c[key],)).fetchone();assert row is not None
                        metadata[identity]=row[col];prior=row[col]
                    assert prior==expect,(identity,'METADATA_MISMATCH')
                physical=canonical({k:c['state'][k]for k in ('positions','velocities','goals')});state=db.execute('SELECT * FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()
                assert state['scenario_uid']==c['scenario_uid']and state['physical_state_json']==physical
                assert state['goals_geometry_json']==canonical(c['scenario_payload'])and state['provenance_json']==canonical(c['state_provenance'])
                if rid in seen:continue
                try:r=result(stage,rid)
                except FileNotFoundError:missing_requests.append(rid);continue
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();assert row is not None,rid
                assert canonical(r['case'])==canonical(c),(rid,'RAW_CASE_DIFF')
                for key in ('scenario_uid','state_uid','eta_uid','controller_uid','seed_key'):
                    if key in row.keys():assert row[key]==r[key]==c[key],(rid,key)
                for key in ('success','collision','timeout','numerical_failure','deadlock'):assert row[key]==r.get(key,0),(rid,key)
                assert row['outcome']==str(r['outcome']).lower(),(rid,'OUTCOME')
                assert row['episode_length']==r['episode_length']or(r['episode_length']==0 and row['episode_length']is None),(rid,'EPISODE_LENGTH')
                assert row['raw_record_hash']==hashlib.sha256(dbm.canonical(r).encode()).hexdigest(),(rid,'DATABASE_RAW_RECORD_HASH')
                assert not row['conflict_quarantined']and not r.get('deadlock',0)
                if r['success']:assert not r['collision']and not r['numerical_failure']
                eta=db.execute('SELECT * FROM eta WHERE eta_uid=?',(c['eta_uid'],)).fetchone();assert eta['canonical_hex']==dbm.eta_identity(c['eta_a'])[2]
                assert digest(r['trace_path'])==r['trace_sha256'],('TRACE_HASH',rid)
                if 'trace_sha256'in row.keys():assert row['trace_sha256']==r['trace_sha256']
                with np.load(r['trace_path'],allow_pickle=False)as z:
                    assert np.array_equal(z['positions'][0],c['state']['positions'])and np.array_equal(z['goals'],c['state']['goals'])
                    assert len(z['actions'])==r['episode_length']
                collision[rid]=collision_types(r,c)
                op=r.get('operational_resume')
                if op:
                    for p,h in op['sources'].items():assert digest(p)==h
                    proof=read(op['validation_receipt']);assert proof['passed']and digest(op['validation_receipt'])==op['validation_receipt_sha256']
                    for p,h in proof['implementation_sources'].items():assert digest(p)==h
                    receipts=set(op['saved_checkpoint_receipts'])|{str(p)for p in(HERE/'operational_checkpoints'/rid).glob('*.receipt.json')}
                    if op.get('resumed_from'):
                        prior=op['resumed_from'];assert digest(prior['receipt'])==prior['receipt_sha256'];receipts.add(prior['receipt'])
                    for p in receipts:
                        saved=read(p);assert saved['scientific_label']is None and saved['case_hash']==uid('checkpoint_case',c)
                        assert saved['sources']==op['sources']and digest(saved['path'])==saved['sha256'];checkpoint_files.add(p)
                seen[rid]=dict(stage=stage,unknown=int(r['numerical_failure']),outcome=r['outcome'],trace_path=r['trace_path'],trace_sha256=r['trace_sha256'])
                timings[rid]=dict(scene=name,controller=ctl,type_code=c['controller_payload']['nominal']['type_code'],seconds=float(r['elapsed_seconds']))
    pending=list((cache_writer.ROOT/'journals'/EXP).glob('*.jsonl'))+list((cache_writer.ROOT/'journals'/EXP).glob('*.tmp'));assert not pending,pending
    write(HERE/'COLLISION_CLASSIFICATION.json',dict(records=collision,source='Simulation result flags, or read-only frozen swept-step criteria on stored collision traces.',counts={k:sum(v[k]for v in collision.values())for k in ('agent','static','moving')}))
    write(HERE/'VERIFIED_RESULT_PROVENANCE.json',seen);write(HERE/'VERIFIED_OUTPUT_SHA256.json',hashes)
    elapsed=np.array([v['seconds']for k,v in timings.items()if k in executed]);events=[json.loads(line)for line in(HERE/'SCHEDULER_EVENTS.jsonl').read_text().splitlines()]
    wall={}
    from datetime import datetime
    for stage in ('capability','main','proposal'):
        begin=[e['utc']for e in events if e['event']=='started'and e.get('stage')==stage]
        end=[e['utc']for e in events if e['event']=='completed'and e.get('stage')==stage]
        wall[stage]=dict(start=begin[0],end=end[-1],seconds=(datetime.fromisoformat(end[-1])-datetime.fromisoformat(begin[0])).total_seconds())
    write(HERE/'ACTUAL_COST.json',dict(unique_rollouts=len(timings),new_execution_count=len(executed-set(missing_requests)),single_cpu_episode_wall_hours_sum=float(elapsed.sum()/3600),
        maximum_episode_seconds=float(elapsed.max()),episode_seconds_quantiles={str(q):float(np.quantile(elapsed,q))for q in (.5,.9,.95,.99)},
        stage_wall=wall,per_request=timings,scope='Single-CPU episode wall times, unique UID counted once; excludes worker import, trace compression and scheduling overhead.'))
    write(HERE/'DB_POSTFLIGHT.json',dict(utc=utc(),passed=True,scope='All completed capability/main/proposal requests and all registered metadata; explicit unfinished requests retained as gaps.',
        stage_requests=counts,unique_requests=len(seen),all_requests_present=not missing_requests,uncompleted_requests=missing_requests,all_identities_and_results_equal=True,
        all_metadata_equal=True,all_cases_exactly_reconstructed=True,all_exact_eta_bytes_equal=True,all_trace_hashes_equal=True,all_database_raw_record_hashes_equal=True,
        no_unmerged_experiment_journals=True,existing_rows_unchanged_at_registration=True,unknowns_retained_without_retry=True,
        operational_checkpoint_receipts=len(checkpoint_files),new_unknowns=sum(v['unknown']for v in seen.values())))

def integrity():
    start=read(HERE/'INTEGRITY_START.json');expected={};protected=ROOT/'diagnostics/cross_scene_prospective_v1'
    def add(path,h):
        p=Path(path).resolve();assert not p.is_relative_to(protected),('RESERVED_FILE_OPEN_ATTEMPT',path)
        prior=expected.setdefault(str(p),h);assert prior==h,('SOURCE_HASH_CHANGED_BETWEEN_READS',path)
    for name in('REUSED_EVIDENCE_SHA256_START.json',):
        for p,h in read(HERE/name)['files'].items():add(p,h)
    for line in(HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():
        r=json.loads(line);add(r['path'],r['sha256'])
    for r in read(HERE/'TOOLING_COPY.receipt.json')['files']:
        add(r['source'],r['source_sha256']);assert sha(r['target'])==r['target_sha256'],r['target']
    r=read(HERE/'AUDIT_SOURCE_COPY.receipt.json');add(r['source'],r['source_sha256'])
    for r in read(HERE/'BACKEND_BUILD.json')['sources']:add(r['source'],r['source_sha256']);assert sha(r['target'])==r['target_sha256']
    assert sha(HERE/'native_resume/cbs_kernel_resume.so')==read(HERE/'BACKEND_BUILD.json')['native_sha256']
    changed={};actual={}
    for p,h in expected.items():
        actual[p]=sha(p)
        if actual[p]!=h:changed[p]=dict(before=h,after=actual[p])
    frozen={}
    for name in('EXECUTION_FROZEN.receipt.json','STATISTICS_FROZEN.receipt.json','TRAINING_FROZEN.receipt.json'):
        for p,h in read(HERE/name)['files'].items():frozen[p]=h
    for receipt,digest in read(HERE/'ALL_FOLDS_FROZEN.receipt.json')['files'].items():
        assert sha(receipt)==digest
        frozen.update(read(receipt)['files'])
    frozen_changed={p:dict(before=h,after=sha(p))for p,h in frozen.items()if sha(p)!=h}
    # Strictly metadata-only: neither read nor hash any reserved scene file.
    meta={}
    for base,dirs,files in os.walk(protected,followlinks=False):
        for name in dirs+files:
            p=Path(base)/name;s=p.stat(follow_symlinks=False);meta[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode)
    prospective_changes={p:dict(before=start['prospective_metadata'].get(p),after=meta.get(p))for p in set(meta)|set(start['prospective_metadata'])if meta.get(p)!=start['prospective_metadata'].get(p)}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:assert not Path(filename).resolve().is_relative_to(protected),('RESERVED_MODULE_IMPORTED',filename)
    prospective_parallel_excepted=bool(prospective_changes)and str(protected)in start['parallel_directories']
    write(HERE/'PROSPECTIVE_METADATA_END.json',dict(metadata=meta,changes=prospective_changes,no_file_content_opened_or_hashed=True,
        registered_parallel_changes_excepted=prospective_parallel_excepted))
    def normalize(status):
        keep=[];skip=[]
        for line in status.splitlines():
            p=line[3:];absolute=str(ROOT/p)
            exc=p.startswith('diagnostics/hetero_scene_pipeline_v1/')or any(absolute.startswith(d.rstrip('/')+'/')for d in start['parallel_directories'])or(line.startswith('??')and p.startswith('docs/prompt_')and p.endswith('.md'))
            (skip if exc else keep).append(line)
        return keep,skip
    repos={};exceptions={}
    for repo,old in start['repositories'].items():
        status=subprocess.check_output(['git','-C',repo,'status','--porcelain']).decode();diff=subprocess.check_output(['git','-C',repo,'diff']);cached=subprocess.check_output(['git','-C',repo,'diff','--cached'])
        before,bs=normalize(old['status']);after,es=normalize(status)
        repos[repo]=dict(status=status,normalized_status_equal=before==after,diff_sha256=hashlib.sha256(diff).hexdigest(),cached_diff_sha256=hashlib.sha256(cached).hexdigest(),
            diff_equal=hashlib.sha256(diff).hexdigest()==old['diff_sha256'],cached_diff_equal=hashlib.sha256(cached).hexdigest()==old['cached_diff_sha256']);exceptions[repo]=dict(before=bs,after=es)
    ok=not changed and not frozen_changed and(not prospective_changes or prospective_parallel_excepted)and all(r['normalized_status_equal']and r['diff_equal']and r['cached_diff_equal']for r in repos.values())
    write(HERE/'REUSED_EVIDENCE_SHA256_END.json',dict(files=actual));write(HERE/'INTEGRITY_END.json',dict(utc=utc(),repositories=repos,exceptions=exceptions,prospective_metadata_unchanged=not prospective_changes,passed=ok))
    write(HERE/'INTEGRITY_COMPARE.json',dict(passed=ok,repositories=repos,exceptions=exceptions,changed_inputs=changed,frozen_changed=frozen_changed,
        source_files=len(expected),frozen_files=len(frozen),reserved_prospective_files_not_opened_or_hashed=True,prospective_metadata_unchanged=not prospective_changes,
        prospective_parallel_changes_excepted=prospective_parallel_excepted))
    if not ok:
        write(HERE/'STOP.receipt.json',dict(utc=utc(),reason='SOURCE_OR_FROZEN_INTEGRITY_FAILURE',changed_inputs=changed,frozen_changed=frozen_changed,prospective_metadata_changes=prospective_changes));raise RuntimeError('SOURCE_OR_FROZEN_INTEGRITY_FAILURE')
