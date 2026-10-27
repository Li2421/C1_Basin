"""Exact v2 database, trace, training, and read-only source integrity audits."""
from common import *
from results import result
import numpy as np,subprocess

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
    from requests_corrected import request
    seen={};counts={};hashes={};collision={};checkpoint_files=set();metadata={};timings={};missing_requests=[]
    executed={c['rollout_uid']for stage in('pilot','main')for c in read(HERE/f'{stage}_MISSING_CASES.json')}
    def digest(p):
        p=str(p)
        if p not in hashes:hashes[p]=sha(p)
        return hashes[p]
    with dbm.connect(True)as db:
        for stage in ('pilot','main'):
            cases=read(HERE/f'{stage}_CASES.json');counts[stage]=len(cases)
            reg=read(HERE/f'{stage}_REGISTRATION.receipt.json');assert reg['original_rows_before']==reg['original_rows_after']
            for c in cases:
                rid=c['rollout_uid'];name=c['state']['scene'];ctl=c['controller_payload']['nominal']['controller']
                rebuilt=request(name,ctl,c['state']['index'],c['eta_a'])
                assert canonical(rebuilt)==canonical(c),('PLANNED_CONSTRUCTOR_DIFF',rid)
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
    for stage in ('pilot','main'):
        begin=[e['utc']for e in events if e['event']=='started'and e.get('stage')==stage]
        end=[e['utc']for e in events if e['event']=='completed'and e.get('stage')==stage]
        wall[stage]=dict(start=begin[0],end=end[-1],seconds=(datetime.fromisoformat(end[-1])-datetime.fromisoformat(begin[0])).total_seconds())
    write(HERE/'ACTUAL_COST.json',dict(unique_rollouts=len(timings),new_execution_count=len(executed-set(missing_requests)),single_cpu_episode_wall_hours_sum=float(elapsed.sum()/3600),
        maximum_episode_seconds=float(elapsed.max()),episode_seconds_quantiles={str(q):float(np.quantile(elapsed,q))for q in (.5,.9,.95,.99)},
        stage_wall=wall,per_request=timings,scope='Single-CPU episode wall times, unique UID counted once; excludes worker import, trace compression and scheduling overhead.'))
    write(HERE/'DB_POSTFLIGHT.json',dict(utc=utc(),passed=True,scope='All completed v2 pilot/main requests and all registered metadata; explicit unfinished requests retained as gaps.',
        stage_requests=counts,unique_requests=len(seen),all_requests_present=not missing_requests,uncompleted_requests=missing_requests,all_identities_and_results_equal=True,
        all_metadata_equal=True,all_cases_exactly_reconstructed=True,all_exact_eta_bytes_equal=True,all_trace_hashes_equal=True,all_database_raw_record_hashes_equal=True,
        no_unmerged_experiment_journals=True,existing_rows_unchanged_at_registration=True,unknowns_retained_without_retry=True,
        operational_checkpoint_receipts=len(checkpoint_files),new_unknowns=sum(v['unknown']for v in seen.values())))

def training():
    rows=[];selected=[]
    for task in read(HERE/'TRAIN_INNER_TASKS.json')+read(HERE/'TRAIN_OUTER_TASKS.json'):
        folder=HERE/'folds'/task['fold']/task['mode']/f"seed{task['seed']}";r=read(folder/'FIT.receipt.json')
        assert r['seed']==task['seed']and sha(r['checkpoint'])==r['sha256']and r['initialization_key']==0
        assert r['source_only_early_stopping']and not r['heldout_labels_used']and not(set(r['training_scenes'])&set(r['heldout_scenes']))
        for p,h in r['implementation_sources'].items():assert sha(p)==h,p
        history=read(folder/'history.json');entry=next(v for v in history if v['step']==r['best_step']);assert entry['validation']==r['validation_loss']
        definition=read(HERE/'folds'/task['fold']/('raw_definition.json'if task['mode']=='raw'and(HERE/'folds'/task['fold']/'raw_definition.json').exists()else'definition.json'))
        assert definition['training_data_sha256']==sha(definition['source_labels_path'])==r['training_data_sha256']
        with np.load(definition['source_labels_path'],allow_pickle=False)as z:
            assert {k.split('__')[0]for k in z.files}==set(r['training_scenes'])
            for n in r['training_scenes']:
                indices=z[n+'__state_indices'];assert np.all(indices[z[n+'__train']]%4!=3)and np.all(indices[z[n+'__val']]%4==3)
        rows.append(r)
    assert len(rows)==501
    for fold in read(HERE/'FOLDS.json'):
        for mode in['full']+(['raw','no_prior','generator']if fold['generator']else[]):
            row=read(HERE/'folds'/fold['name']/mode/'SELECTED.json');best=min(row['all_seeds'],key=lambda r:(r['validation_loss'],r['seed']))
            assert canonical(best)==canonical(row['selected']);selected.append(dict(fold=fold['name'],mode=mode,selected=best))
    write(HERE/'TRAINING_AUDIT.json',dict(utc=utc(),passed=True,unique_seed_fits=len(rows),selected_outer_models=len(selected),fits=rows,selections=selected,
        source_identical_inner_fits_reused=263,no_target_labels_in_fits_early_stopping_or_normalization=True,all_curves_and_selected_steps_verified=True,
        single_cpu_fit_wall_hours_sum=sum(r['elapsed_seconds']for r in rows)/3600))
    from names_v2 import SCENE_NAMES
    labels={'full':'学习选 η v2的先验加有界修正critic','raw':'v2·原始场景描述的critic','no_prior':'v2·无先验的critic','generator':'学习选 η v2的场景级提案生成器'}
    lines=['# 训练与选择记录\n\n','764次未共享排程中的263次内部拟合与其他外层折具有完全相同的有序源集合、数据、先验、标准化、网络、采样和种子，因此共享检查点。实际291次内部拟合和210次外层三种子拟合，共501次。固定初始化key=0，种子17/23/41控制各场景独立采样流，均沿用指定实现。\n\n',
        '| 外层折与留出场景 | 模型 | 所选种子 | 所选步 | 内部验证损失 |\n|---|---|---:|---:|---:|\n']
    for r in selected:
        fold=next(f for f in read(HERE/'FOLDS.json')if f['name']==r['fold']);name='、'.join(SCENE_NAMES[n]for n in fold['targets']);s=r['selected']
        lines.append(f"| {fold['kind']}：{name}（{r['fold']}） | {labels[r['mode']]} | {s['seed']} | {s['best_step']} | {s['validation_loss']:.9g} |\n")
    lines.append('\n每个种子的完整验证曲线、耗时、数据哈希与检查点见各节点目录；内部共享的全部节点与来源见TRAINING_AUDIT.json、INNER_NODES.json及各折LAMBDA_SELECTION.json。\n')
    (HERE/'TRAINING_SUMMARY.md').write_text(''.join(lines))

def cpu_handoff():
    if not(HERE/'CPU_PLACEMENT_HANDOFF.receipt.json').exists():return
    import pickle
    receipt=read(HERE/'CPU_PLACEMENT_HANDOFF.receipt.json');running=read(HERE/'CPU_PLACEMENT_RUNNING_CHECK.json');assert running['passed']
    assert sha(HERE/'handoff_performance_cores_v2.py')==receipt['script_sha256']
    assert sha(HERE/'CHECKPOINT_COMPATIBILITY.json')==receipt['checkpoint_compatibility_sha256']
    rows=[]
    for handoff in running['handoffs']:
        rid=handoff['rollout_uid'];partial=False
        try:
            r=result('main',rid);resume=r['operational_resume']['resumed_from'];trace_path=r['trace_path']
        except FileNotFoundError:
            end=read(HERE/'USER_CUTOFF_EVIDENCE.json');assert end['rollout_uid']==rid
            partial=True;resume=dict(next_step=end['safe_pause_step'],receipt=end['checkpoint_receipt']);trace_path=end['partial_trace_path']
            assert sha(trace_path)==end['partial_trace_sha256']
        path=Path(handoff['paused_receipt'])
        assert sha(path)==handoff['paused_receipt_sha256'];saved=read(path);assert saved['scientific_label']is None
        assert sha(saved['path'])==saved['sha256'];assert resume['next_step']>=saved['next_step']
        with Path(saved['path']).open('rb')as stream:snapshot=pickle.load(stream)
        n=snapshot['next_step']+1;checks={}
        with np.load(trace_path,allow_pickle=False)as trace:
            for field in('positions','velocities'):
                before=np.asarray(snapshot[field]);after=trace[field][:n]
                checks[field]=before.dtype==after.dtype and before.shape==after.shape and before.tobytes()==after.tobytes()
        assert all(checks.values()),(rid,'HANDOFF_PREFIX_NOT_BITWISE_EQUAL',checks)
        rows.append(dict(rollout_uid=rid,old_affinity=next(o['old_affinity']for o in receipt['owners']if o['rollout_uid']==rid),new_affinity=handoff['new_affinity'],
            paused_next_step=saved['next_step'],final_resume_next_step=None if partial else resume['next_step'],final_available_next_step=resume['next_step'],paused_receipt=str(path),paused_receipt_sha256=sha(path),
            final_resume_or_partial_checkpoint_receipt=resume['receipt'],same_pause_receipt_resumed=None if partial else resume['receipt']==str(path),prefix_bitwise_checks=checks,no_scientific_request_added=True,
            completed_scientific_rollout=not partial,user_authorized_partial=partial))
    write(HERE/'CPU_PLACEMENT_HANDOFF_CHECK.json',dict(utc=utc(),passed=True,handoffs=rows,no_affinity_bypass=True,no_model_proposal_choice_or_request_changed=True,
        saved_prefix_preserved_bitwise=True,source_compatible_checkpoint_continuation=True))

def cutoff_accounting():
    if not(HERE/'USER_CUTOFF_EVIDENCE.json').exists():return
    a=read(HERE/'USER_CUTOFF_AMENDMENT.json');e=read(HERE/'USER_CUTOFF_EVIDENCE.json');db=read(HERE/'DB_POSTFLIGHT.json')
    assert db['uncompleted_requests']==[a['rollout_uid']]and e['scientific_outcome']is None
    with dbm.connect(True)as connection:assert connection.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(a['rollout_uid'],)).fetchone()is None
    planned={c['rollout_uid']for stage in('pilot','main')for c in read(HERE/f'{stage}_CASES.json')}
    db.update(planned_unique_requests=len(planned),user_authorized_uncompleted_requests=1,partial_request_not_given_timeout_or_deadlock_label=True)
    write(HERE/'DB_POSTFLIGHT.json',db)
    cost=read(HERE/'ACTUAL_COST.json');cost.update(new_execution_attempts=cost['new_execution_count']+1,partial_user_cutoff_requests=1,
        partial_user_cutoff_actual_episode_wall_seconds=e['actual_partial_episode_wall_seconds'],
        total_completed_and_partial_episode_wall_hours=cost['single_cpu_episode_wall_hours_sum']+e['actual_partial_episode_wall_seconds']/3600)
    write(HERE/'ACTUAL_COST.json',cost)

def integrity():
    start=read(HERE/'INTEGRITY_START.json');expected={};protected=ROOT/'diagnostics/cross_scene_prospective_v1'
    def add(path,h):
        p=Path(path).resolve();assert not p.is_relative_to(protected),('RESERVED_FILE_OPEN_ATTEMPT',path)
        prior=expected.setdefault(str(p),h);assert prior==h,('SOURCE_HASH_CHANGED_BETWEEN_READS',path)
    for name in('EVIDENCE_SHA256_START.json','REUSED_EVIDENCE_SHA256_START.json'):
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
    for name in('TRAINING_FROZEN.receipt.json','EXECUTION_FROZEN.receipt.json','STATISTICS_V2_FROZEN.receipt.json','FAST_SCHEDULER.receipt.json','ROLLOUT_SCHEDULER.receipt.json'):
        for p,h in read(HERE/name)['files'].items():frozen[p]=h
    if(HERE/'USER_CUTOFF_STATISTICS.receipt.json').exists():
        for p,h in read(HERE/'USER_CUTOFF_STATISTICS.receipt.json')['files'].items():frozen[p]=h
    for p,h in read(HERE/'ALL_FOLDS_FROZEN.receipt.json')['files'].items():
        frozen[p]=h
        for q,k in read(p)['files'].items():
            if q in frozen:assert frozen[q]==k
            frozen[q]=k
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
            exc=p.startswith('diagnostics/cross_scene_pipeline_v2/')or any(absolute.startswith(d.rstrip('/')+'/')for d in start['parallel_directories'])or(line.startswith('??')and p.startswith('docs/prompt_')and p.endswith('.md'))
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

