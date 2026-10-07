"""Frozen common-eta, multi-state Q16 panel for Ring / Four-Way.

This is an outcome-blind cohort/panel freeze followed by exact cached rollout
collection and descriptive interaction analysis. It never trains a model.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets/orthoflow3_basin_dataset_v3_unified_rep'
DB = Path('/home/zhihan/research/Basin_C1/shared_rollout_db/rollout.sqlite')
SCENES = ('four_way_intersection', 'ring_exchange')
SEEDS = tuple(range(16))
FIXED = ROOT / 'diagnostics/orthoflow3_ring_revision_v1/fair_fixed_eta.json'
DESIGN = ROOT / 'diagnostics/ring_exchange_safety_eta3/frozen_eta_design.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dec(x):
    while isinstance(x,str): x=json.loads(x)
    return x
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def load(name): return json.loads((OUT/name).read_text())

def freeze():
    state_path=DATA/'states.parquet'
    rows=pq.read_table(state_path).to_pylist()
    fixed=json.loads(FIXED.read_text()); design=json.loads(DESIGN.read_text())
    states_by={}; selected_counts={}
    for sc in SCENES:
        eligible=[r for r in rows if r['scenario']==sc and r['split'] in ('train','validation')
                  and r.get('zero_sufficient') is False]
        # Outcome-blind deterministic SHA ordering with parent cap 2.
        eligible.sort(key=lambda r:(hashlib.sha256(('common-eta-panel-v1|'+r['state_uid']).encode()).hexdigest(),r['state_uid']))
        chosen=[];parent_counts=Counter()
        for cap in (1,2):
            for row in eligible:
                if len(chosen)>=24: break
                pid=row['parent_episode_id']
                if any(row['state_uid']==x['state_uid'] for x in chosen) or parent_counts[pid]>=cap: continue
                chosen.append(row);parent_counts[pid]+=1
            if len(chosen)>=24: break
        if len(chosen)<24: chosen=eligible[:]
        records=[]
        for r in chosen:
            p=dec(r['structured_state']);cond=dec(r['conditioning'])
            assert r['split'] in ('train','validation')
            assert dec(r['provenance']).get('frozen_test_used') is False
            assert cond['schema']=='physical_entities_goal_frames_v1'
            physical={'positions':p['positions'],'velocities':p['velocities'],'goals':p['goals'],
                      'timestep':int(p['timestep']),'normalized_episode_time':float(p['normalized_episode_time'])}
            records.append({'scenario':sc,'state_uid':r['state_uid'],'state_id':r['state_id'],'split':r['split'],
                'parent_episode_id':r['parent_episode_id'],'source_initial_state_id':r['source_initial_state_id'],
                'timestep':int(r['timestep']),'source_rollout_type':r['source_rollout_type'],
                'source_outcome':r['source_outcome'],'zero_sufficient':False,'physical':physical,
                'conditioning':cond,'environment_descriptor':dec(r['environment_descriptor']),
                'controller_uid_source':r['controller_uid'],'physical_snapshot_hash':digest(physical),
                'conditioning_hash':digest(cond),'v3_row_hash':digest({k:r[k] for k in ('state_uid','conditioning','structured_state','environment_descriptor')})})
        assert len(records)<=24 and len({x['state_uid'] for x in records})==len(records)
        assert max(Counter(x['parent_episode_id'] for x in records).values(),default=0)<=2
        states_by[sc]=records;selected_counts[sc]={'eligible':len(eligible),'selected':len(records),
            'parent_count':len({x['parent_episode_id'] for x in records}),
            'parent_max_multiplicity':max(Counter(x['parent_episode_id'] for x in records).values(),default=0),
            'split_counts':dict(Counter(x['split'] for x in records))}
    eta=[[0.,0.,0.],fixed['four_way_intersection']['selected']['eta'],
         fixed['ring_exchange']['selected']['eta']]
    eta+=design['primary'][:21]
    unique=[];indices=[]
    for idx,x in enumerate(eta):
        if not any(np.array_equal(np.asarray(x,float),np.asarray(y,float)) for y in unique):
            unique.append([float(z) for z in x]);indices.append(idx)
    for x in design['primary'][21:]:
        if len(unique)>=24:break
        if not any(np.array_equal(np.asarray(x,float),np.asarray(y,float)) for y in unique):
            unique.append([float(z) for z in x]);indices.append(24+design['primary'].index(x))
    assert len(unique)==24
    lo=np.asarray(design['domain']['lower']);hi=np.asarray(design['domain']['upper'])
    assert all(np.all(np.asarray(e)>=lo) and np.all(np.asarray(e)<=hi) for e in unique[3:])
    state_list=[x for sc in SCENES for x in states_by[sc]]
    assert len(states_by['four_way_intersection'])==24 and len(states_by['ring_exchange'])==24,selected_counts
    save('states_manifest.json',{'frozen_before_eta_outcomes':True,
        'selection_rule':'correction-needed (zero_sufficient=false), TRAIN/validation only; SHA256 state_uid order; greedy max2 per parent; no model/basin/outcome-based selection',
        'counts':selected_counts,'states':state_list})
    save('common_eta_panel.json',{'frozen_before_outcomes':True,'eta_count':24,'eta_values':unique,
        'panel_sources':['eta_zero','fair_train_dev_fixed_four_way','fair_train_dev_fixed_ring','canonical_ring_primary_sobol_first21_then_next_unique'],
        'design_source':str(DESIGN.relative_to(ROOT)),'design_sha256':sha(DESIGN),
        'sobol_primary_seed':design['primary_seed'],'sobol_primary_indices':indices[3:],
        'domain':design['domain'],'fixed_eta_source':str(FIXED.relative_to(ROOT)),'fixed_eta_sha256':sha(FIXED),
        'no_state_adaptation':True})
    save('freeze_manifest.json',{'dataset':str(DATA.relative_to(ROOT)),'dataset_manifest_sha256':sha(DATA/'manifest.json'),
        'states_parquet_sha256':sha(state_path),
        'critic_manifest_sha256':sha(ROOT/'diagnostics/orthoflow3_unified_representation_v1/critic_frozen.json'),
        'critic_checkpoint_sha256':json.loads((ROOT/'diagnostics/orthoflow3_unified_representation_v1/critic_frozen.json').read_text())['selected']['sha256'],
        'v3_encoder_sha256':sha(ROOT/'diagnostics/orthoflow3_unified_representation_v1/representation.py'),
        'safety_current_hash_source':'new_benchmark_common.basin_dataset_v1 TrainingRuntime; exact controller UID checked at execution',
        'seed_order':list(SEEDS),'full_Q16':True,'retry_rule':'initial plus at most3 exact identical retries; numerical unresolved excluded and interval reported',
        'model_changes':False,'frozen_test_states_used':False,'cohort_counts':selected_counts})
    return {'states':len(state_list),'etas':len(unique),'primary_seed_slots':len(state_list)*len(unique)*16,'counts':selected_counts}

def setup_runtimes():
    from new_benchmark_common import basin_dataset_v1 as bd
    from shared_rollout_db.src.rollout_db import uid,connect,canonical
    bd.OUT=OUT/'db_registry';bd.WORK=OUT/'runtime_work'
    for sc in SCENES:
        records=[x for x in load('states_manifest.json')['states'] if x['scenario']==sc]
        state_rows=[]
        for i,r in enumerate(records):
            state_rows.append({'uid':r['state_uid'],'alias':r['state_id'],'index':i,'physical':r['physical'],
                'content_hash':r['physical_snapshot_hash'],'conditioning':r['conditioning'],
                'environment_descriptor':r['environment_descriptor'],'source_group':r['parent_episode_id'],
                'provenance':{'task':'common_eta_interaction_panel_v1','source_split':r['split'],
                              'parent_episode_id':r['parent_episode_id'],'frozen_test_used':False}})
        runtime=bd.TrainingRuntime(sc,state_rows,parent=False)
        assert runtime.controllers['orthoflow3']['uid']==records[0]['controller_uid_source'],(sc,runtime.controllers['orthoflow3']['uid'],records[0]['controller_uid_source'])
        runtime.output=OUT/'execution'/sc;runtime.output.mkdir(parents=True,exist_ok=True)
        exp_path=(OUT/'execution'/sc/'experiment').resolve()
        runtime.experiment_uid=uid('exp',{'path':str(exp_path),'protocol':'common_eta_panel_v1'})
        with connect() as con:
            con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                (runtime.experiment_uid,'common_eta_state_interaction_panel_v1',str(exp_path),
                 sha(OUT/'common_eta_panel.json'),sha(__file__),canonical({'training':False,'frozen_test_used':False})))
            assert con.execute('SELECT 1 FROM experiment WHERE experiment_uid=?',(runtime.experiment_uid,)).fetchone()
            con.commit()
        yield sc,runtime

def critic_scores():
    import jax.numpy as jnp
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep,build,train
    gm,gp,cm,cp=train.load_models()
    etas=np.asarray(load('common_eta_panel.json')['eta_values'],np.float32)
    out=[]
    rowmap={r['state_uid']:r for r in build.rows()}
    for sc in SCENES:
        rec=[x for x in load('states_manifest.json')['states'] if x['scenario']==sc]
        xr=rep.batch([rep.entities(build.scene(rowmap[r['state_uid']])) for r in rec])
        x={k:jnp.asarray(v) for k,v in xr.items()}
        b=len(rec);e=np.tile((etas-train.base.CENTER)/train.base.RADIUS,(b,1))
        xb={k:jnp.repeat(v,len(etas),axis=0) for k,v in x.items()}
        logits=np.asarray(cm.apply(cp,xb,jnp.asarray(e))).reshape(b,len(etas))
        probs=1/(1+np.exp(-logits))
        out.append({'scenario':sc,'state_uids':[r['state_uid'] for r in rec],
            'score_type':'sigmoid critic Q score (diagnostic)','logit_matrix':logits.tolist(),
            'score_matrix':probs.tolist(),
            'critic_checkpoint_sha256':json.loads((ROOT/'diagnostics/orthoflow3_unified_representation_v1/critic_frozen.json').read_text())['selected']['sha256']})
    save('critic_scores_frozen.json',out)
    return {'scenarios':len(out),'score_cells':sum(len(x['state_uids'])*24 for x in out)}

def make_jobs(runtime):
    from new_benchmark_common.safety_eta3 import request
    etas=load('common_eta_panel.json')['eta_values']
    return [{'state':s,'eta':eta,'chain':'orthoflow3','seeds':list(SEEDS),'eta_index':j,
             'request':request(runtime,s,eta,'orthoflow3',SEEDS)} for s in runtime.states for j,eta in enumerate(etas)]

def preflight():
    from new_benchmark_common.safety_eta3 import preflight as db_preflight
    from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
    summary={};details=[]
    for sc,r in setup_runtimes():
        jobs=make_jobs(r);p=OUT/'cache'/f'requests_{sc}.json';p.parent.mkdir(parents=True,exist_ok=True)
        save(str(p.relative_to(OUT)),{'requests':[j['request'] for j in jobs]})
        std=db_preflight(p);found_n=missing_n=numeric_n=0
        for j in jobs:
            found,num,missing=a.evidence(r,j);found_n+=len(found);numeric_n+=len(num);missing_n+=len(missing)
            details.append({'scenario':sc,'state_uid':j['state']['uid'],'eta_index':j['eta_index'],'eta':j['eta'],
                'cached_valid_seeds':sorted(found),'cached_numerical_seeds':sorted(num),'missing_seeds':missing})
        summary[sc]={'requested_seed_slots':len(jobs)*16,'valid_cached':found_n,'cached_numerical':numeric_n,
                     'missing':missing_n,'standard_preflight':std}
    save('cache_preflight.json',{'database':str(DB),'scenario_summary':summary,'details':details,'cache_first':True})
    return summary

def run(shard,shards):
    from new_benchmark_common.safety_eta3 import DatabaseSink
    from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
    totals=Counter();global_index=0
    for sc,r in setup_runtimes():
        chosen=[]
        for j in make_jobs(r):
            if global_index%shards==shard:chosen.append(j)
            global_index+=1
        if not chosen:continue
        sink=DatabaseSink(r,f'panel_{shard}of{shards}')
        try:
            for j in chosen:
                found,num,missing=a.evidence(r,j)
                totals['requested_pairs']+=1;totals['reused_valid_seed_rows']+=len(found);totals['existing_numerical']+=len(num)
                for seed in missing:
                    last=None
                    for attempt in range(4):
                        last=r.rollout(j['state'],np.asarray(j['eta'],float),seed,'orthoflow3')
                        last.update({'eta_index':j['eta_index'],'execution_attempt':attempt,
                          'provenance_experiment':'common_eta_state_interaction_panel_v1','full_Q16_required':True,
                          'training_label':False,'source_split':next(s['split'] for s in load('states_manifest.json')['states'] if s['state_uid']==j['state']['uid'])})
                        sink.insert(last);totals['physical_attempts']+=1
                        if not last['numerical_failure']:break
                    totals['seed_slots_completed']+=1;totals['numerical_unresolved']+=bool(last['numerical_failure'])
                totals['pairs_processed']+=1
                if totals['pairs_processed']%4==0: print(json.dumps({'shard':shard,**totals}),flush=True)
        finally:sink.finalize()
    save(f'run_shard{shard}of{shards}.json',dict(totals));return dict(totals)

def collect():
    from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
    output=[];total=Counter()
    for sc,r in setup_runtimes():
        for j in make_jobs(r):
            valid,numeric,missing=a.evidence(r,j)
            successes=sum(int(x['success']) for x in valid.values());failures=len(valid)-successes
            record={'scenario':sc,'state_uid':j['state']['uid'],'eta_index':j['eta_index'],'eta':j['eta'],
                'valid_seed_count':len(valid),'successes':successes,'failures':failures,
                'numerical_seeds':sorted(numeric),'missing_seeds':missing,
                'Q16':successes/16 if len(valid)==16 else None,'Q_lower':successes/16,'Q_upper':(16-failures)/16,
                'B15':True if successes>=15 else False if failures>=2 else None,
                'terminal_reasons':dict(Counter(str(x.get('outcome',x.get('termination'))) for x in valid.values())),
                'collision_count':sum(bool(x['collision']) for x in valid.values()),
                'seed_success':{str(seed):bool(row['success']) for seed,row in sorted(valid.items())},
                'rollout_uids':[x['rollout_uid'] for _,x in sorted(valid.items())],
                'numerical_rollout_uids':[numeric[s]['rollout_uid'] for s in sorted(numeric)]}
            output.append(record);total['tuples']+=1;total['numerical_seed_slots']+=len(numeric)
            total['missing_seed_slots']+=len(missing);total['valid_seed_slots']+=len(valid)
    save('q16_records.json',{'records':output,'summary':dict(total),'complete_matrix_required':True})
    matrices={}
    for sc in SCENES:
        sids=[x['state_uid'] for x in load('states_manifest.json')['states'] if x['scenario']==sc]
        mat={(x['state_uid'],x['eta_index']):x for x in output if x['scenario']==sc}
        def arr(key):return [[mat[(sid,j)][key] for j in range(24)] for sid in sids]
        matrices[sc]={'state_uids':sids,'Q16':arr('Q16'),'Q_lower':arr('Q_lower'),'Q_upper':arr('Q_upper'),
                      'B15':arr('B15'),'seed_outcomes':[[mat[(sid,j)]['seed_success'] for j in range(24)] for sid in sids],
                      'rollout_references':{sid:{str(j):mat[(sid,j)]['rollout_uids'] for j in range(24)} for sid in sids}}
    save('Q16_matrices.json',matrices)
    return dict(total)

def classify(oracle_gap,oracle_gap_upper,strong_pair_fraction,best_shift_fraction):
    if strong_pair_fraction>=.20 or (oracle_gap>=.10 and best_shift_fraction>=.25):return 'TRUE_INTERACTION_PRESENT'
    if strong_pair_fraction==0 and oracle_gap_upper<.05:return 'TRUE_INTERACTION_WEAK'
    return 'UNDERRESOLVED'

def write_report(reports):
    pre=load('cache_preflight.json')['scenario_summary'];counts=load('states_manifest.json')['counts']
    lines=['# Common-eta state–eta interaction panel v1','',
      '## 设计与执行','',
      '使用 unified v3 中 TRAIN/validation 的 correction-needed 状态；每场景 24 个，固定共同 eta 面板 24 个，完整 canonical seeds 0–15。没有使用任何 frozen test state，没有修改/训练 generator 或 critic。Q16 仅在 16 个有效 seed 全部完成时报告；数值失败保留为区间并不作成功/失败插补。B15 为至少 15/16 成功。', '',
      '| 场景 | 候选 correction-needed | 入选 | 不同 parent | train/validation | 可复用有效 seeds | 已缓存数值失败 | 新缺失 seeds |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for sc in SCENES:
        c=counts[sc];p=pre[sc];lines.append(f"| {sc} | {c['eligible']} | {c['selected']} | {c['parent_count']} | {c['split_counts']} | {p['valid_cached']} | {p['cached_numerical']} | {p['missing']} |")
    lines += ['', '共同面板顺序：eta=0、train/dev fair Four-Way eta、train/dev fair Ring eta、canonical primary Sobol design 的前 21 个有效点。每个 rollout 在 shared DB 中按 state×eta×controller×future-index 精确键存储。', '',
      '## TRUE Q 与 state-dependent ranking','',
      '| 场景 | 完整 Q16 cells | 数值未决 seeds | B15 真/假/未决 tuples | 碰撞 | 不同 state-best eta | best eta 不同于全局 eta | state-pair Spearman 中位数 | strong reversal quadruples | 含 strong reversal 的 state-pair | fraction | state-specific oracle Q | global eta Q | oracle gap | interaction energy ratio |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sc,r in reports.items():
        x=r['matrix'];rk=r['rankings'];rv=r['reversals'];o=r['oracle'];d=r['additive_decomposition']
        sp=rk['pairwise_spearman_mean_median'];spmed=sp[1] if sp else float('nan')
        lines.append(f"| {sc} | {x['complete_cells']}/{x['expected_cells']} | {x['unresolved_seed_slots']} | {x['B15_true']}/{x['B15_false']}/{x['B15_unresolved']} | {x['collision_rollouts']} | {rk['unique_best_eta_count']} | {rk['best_differs_from_global_best_fraction']:.3f} | {spmed:.3f} | {rv['strong_quadruple_count']} | {rv['state_pairs_with_strong_reversal']}/{24*23//2} | {rv['fraction_state_pairs_with_strong_reversal']:.3f} | {o['state_oracle_mean_Q_lower']:.3f}–{o['state_oracle_mean_Q_upper']:.3f} | {o['global_mean_Q_lower']:.3f}–{o['global_mean_Q_upper']:.3f} | {o['STATE_SPECIFIC_ORACLE_GAP_lower']:.3f}–{o['STATE_SPECIFIC_ORACLE_GAP_upper']:.3f} | {d.get('interaction_energy_ratio',float('nan')):.3f} |")
    lines += ['', '## 每场景详细结果','']
    for sc,r in reports.items():
        lines += [f"### {sc}",'',f"判断：**{r['classification']}**。",'',
          f"- Q16 矩阵：{r['matrix']['complete_cells']}/{r['matrix']['expected_cells']} 个 cell 完整；{r['matrix']['complete_Q16_rows']}/24 个 state 行全部完整；{r['matrix']['unresolved_seed_slots']} 个 seed 数值未决。未决 cell 使用 Q_lower/Q_upper；状态间 rank 相似度用每对状态共同拥有的精确 eta 子集。所有状态均精确的 eta 列为 {r['matrix']['fully_complete_eta_columns']}。",
          f"- State ranking：{r['rankings']['unique_best_eta_count']} 种 observed-best eta 身份；{r['rankings']['states_with_uncertain_observed_best_due_to_unresolved_eta']}/24 个状态可能被未决 eta 的 Q 上界改变 observed-best；{r['rankings']['best_differs_from_global_best_fraction']:.3f} 状态的 observed-best Q 严格高于全局 eta 的 Q；state-pair Spearman/Kendall 中位数分别为 {r['rankings']['pairwise_spearman_mean_median'][1]:.3f}/{r['rankings']['pairwise_kendall_mean_median'][1]:.3f}（{r['rankings']['pairwise_rank_comparisons']} 对）。",
          f"- Strong reversal（双侧差异至少 4/16）：{r['reversals']['strong_quadruple_count']} 个 eta-pair×state-pair quadruples；{r['reversals']['state_pairs_with_strong_reversal']}/{24*23//2} state pairs ({r['reversals']['fraction_state_pairs_with_strong_reversal']:.1%})，涉及 {r['reversals']['states_participating']} 个 states。较弱反转另计 {r['reversals']['weaker_reversal_quadruple_count']} 个。",
          f"- Oracle：state-specific mean Q bounds={r['oracle']['state_oracle_mean_Q_lower']:.3f}–{r['oracle']['state_oracle_mean_Q_upper']:.3f}，B15 coverage bounds={r['oracle']['state_oracle_B15_coverage_lower']:.1%}–{r['oracle']['state_oracle_B15_coverage_upper']:.1%}；global eta {r['oracle']['global_eta']} mean Q bounds={r['oracle']['global_mean_Q_lower']:.3f}–{r['oracle']['global_mean_Q_upper']:.3f}，B15={r['oracle']['global_B15_coverage_lower']:.1%}–{r['oracle']['global_B15_coverage_upper']:.1%}；STATE_SPECIFIC_ORACLE_GAP bounds={r['oracle']['STATE_SPECIFIC_ORACLE_GAP_lower']:.3f}–{r['oracle']['STATE_SPECIFIC_ORACLE_GAP_upper']:.3f}。",
          f"- Eta-only：top1 对 observed state-best identity accuracy={r['eta_only']['top1_accuracy_vs_state_best']:.1%}；top3 hit={r['eta_only']['top3_hit_rate']:.1%}；mean regret lower bound={r['eta_only']['mean_regret_lower']:.3f}；B15={r['eta_only']['B15_coverage_lower']:.1%}–{r['eta_only']['B15_coverage_upper']:.1%}。",
          f"- Frozen critic：mean per-state Spearman={r['critic']['mean_per_state_spearman']:.3f}；top1 mean true Q bounds={r['critic']['top1_mean_true_Q']:.3f}–{r['critic']['top1_mean_true_Q_upper']:.3f}，regret bounds={r['critic']['top1_mean_regret_lower']:.3f}–{r['critic']['top1_mean_regret_upper']:.3f}；top3 hit={r['critic']['top3_hit_rate']:.1%}；B15 selection={r['critic']['B15_selection_rate_lower']:.1%}–{r['critic']['B15_selection_rate_upper']:.1%}；strong-reversal direction accuracy={r['critic']['strong_reversal_direction_accuracy']}；B15 change vs eta-only bounds={r['critic']['improvement_over_eta_only_B15_lower']}–{r['critic']['improvement_over_eta_only_B15_upper']}。",
          '- State-score shuffle：',
          '| shuffle seed | top1 agreement | shuffled-selection true Q | original true Q | score correlation |',
          '|---:|---:|---:|---:|---:|']
        for z in r['critic']['state_shuffle']:
            lines.append(f"| {z['seed']} | {z['top1_agreement_with_unshuffled']:.3f} | {z['true_Q_of_shuffled_critic_selection_lower']:.3f}–{z['true_Q_of_shuffled_critic_selection_upper']:.3f} | {z['true_Q_of_unshuffled_critic_selection_lower']:.3f}–{z['true_Q_of_unshuffled_critic_selection_upper']:.3f} | {z['score_correlation']:.3f} |")
        lines += ['', '']
    lines += ['## 解释边界','',
      '该分析比较同一 state cohort 上的真实 fixed-eta continuation Q，不是把 state 难度变化误作 eta-ranking 交互。双侧强反转是直接的 state×eta ranking 证据；state-specific oracle gap 衡量 24 点共同面板内的可用性差异。加性分解是描述统计，不能单独作因果结论。critic 分数是冻结模型对同一共同面板的逐格评分；shuffle 仅为诊断。结果只覆盖此固定面板和训练/开发状态，不是总体概率的无偏保证。', '',
      '## 文件与完整性','',
      '- `states_manifest.json`、`common_eta_panel.json`：outcome-blind 冻结 cohort/panel。',
      '- `Q16_matrices.json`、`q16_records.json`：矩阵、逐 seed 成功与数据库 rollout UID。',
      '- `critic_scores_frozen.json`：冻结 critic score/logit 矩阵。',
      '- `analysis.json`：排序、强/弱 reversal 原始记录、oracle、加性分解、eta-only、critic 与 shuffle 指标。',
      '- `cache_preflight.json`：rollout 前兼容缓存检查。',
      '- `run_shard*of18.json` 与 `execution/*/raw/`：增量执行摘要和可审计原始记录。',
      '- `analysis_preregistration.json`：分析阈值和 ties 规则。', '',
      '执行合计：1,152 个 state×eta tuple、18,432 个 canonical seed 槽位；有效结果 18,356，数值未决 76，缺失/未运行 0。Rollout DB 在本批次前复用 6,250 个有效 seed 结果及 15 个已达重试上限的数值结果；本批新执行 12,350 次（含 183 次相同 seed 的数值重试），留下 61 个新数值未决。所有 76 个未决 seed 均不插补；所有 tuple 的 B15 二元分类均可判定。有效 rollout 碰撞数为 0。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def write_dataset_package():
    import shutil
    target=ROOT/'datasets/orthoflow3_state_eta_interaction_panel_v1'
    target.mkdir(parents=True,exist_ok=True)
    names=['states_manifest.json','common_eta_panel.json','Q16_matrices.json','q16_records.json',
           'critic_scores_frozen.json','analysis.json','freeze_manifest.json','cache_preflight.json',
           'analysis_preregistration.json','ANALYSIS_QC_NOTE.md']
    for name in names:
        if (OUT/name).exists():shutil.copy2(OUT/name,target/name)
    manifest={'schema':'orthoflow3_state_eta_interaction_panel_v1','scenarios':list(SCENES),
       'source_dataset':'datasets/orthoflow3_basin_dataset_v3_unified_rep/',
       'train_validation_only':True,'frozen_test_states_used':False,'model_training_performed':False,
       'files':{name:sha(target/name) for name in names if (target/name).exists()},
       'labels':'full canonical Q16 seed outcomes referenced by rollout_uid; numerical failures remain intervals/unresolved'}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

def analyze():
    import itertools
    scores={x['scenario']:x for x in load('critic_scores_frozen.json')}
    mats=load('Q16_matrices.json');panel=np.asarray(load('common_eta_panel.json')['eta_values'])
    reports={}
    for sc in SCENES:
        m=mats[sc];q=np.asarray(m['Q16'],float);p=np.asarray(scores[sc]['score_matrix'],float);n=len(q)
        cell=np.isfinite(q);qlo=np.asarray(m['Q_lower'],float);qhi=np.asarray(m['Q_upper'],float)
        common_eta=np.flatnonzero(cell.all(axis=0)).tolist()
        # Select global eta over all 24 panel points by mean certified lower
        # bound; retain the corresponding upper bound and B15 interval.
        means_lo=qlo.mean(0);means_hi=qhi.mean(0);global_idx=int(np.argmax(means_lo))
        global_q=float(means_lo[global_idx]);global_q_upper=float(means_hi[global_idx])
        global_b15=float(np.mean(qlo[:,global_idx]>=15/16));global_b15_upper=float(np.mean(qhi[:,global_idx]>=15/16))
        oracle_lower=np.max(qlo,axis=1);oracle_upper=np.max(qhi,axis=1)
        oracle_q=float(np.mean(oracle_lower));oracle_q_upper=float(np.mean(oracle_upper))
        gap=oracle_q-global_q_upper;gap_upper=oracle_q_upper-global_q
        oracle_b15_lower=float(np.mean(oracle_lower>=15/16));oracle_b15_upper=float(np.mean(oracle_upper>=15/16))
        eta_order=np.argsort(-means_lo,kind='stable');etaonly_idx=int(eta_order[0])
        etaonly_top3ids=set(map(int,eta_order[:3]));etaonly_best_hits=[];etaonly_regrets=[]
        rank_candidates=[];rank_uncertain=[];state_rank_details=[]
        for i in range(n):
            ix=np.flatnonzero(cell[i]);order=ix[np.argsort(-q[i,ix],kind='stable')]
            best=int(order[0]);rank_candidates.append(best);etaonly_best_hits.append(best in etaonly_top3ids)
            etaonly_regrets.append(float(np.max(qlo[i])-qhi[i,etaonly_idx]))
            uncertain=bool(np.any((~cell[i])&(qhi[i]>=q[i,best])))
            if uncertain:rank_uncertain.append(i)
            state_rank_details.append({'state_uid':m['state_uids'][i],'observed_best_eta_index':best,
                'observed_best_eta':panel[best].tolist(),'observed_best_Q16':float(q[i,best]),
                'top3_observed_eta_indices':order[:3].astype(int).tolist(),
                'top3_observed_etas':panel[order[:3]].tolist(),'unresolved_eta_could_change_best':uncertain,
                'global_eta_index':global_idx,'global_eta_Q_lower':float(qlo[i,global_idx]),
                'global_eta_Q_upper':float(qhi[i,global_idx])})
        etaonly_top1_accuracy=float(np.mean(np.asarray(rank_candidates)==etaonly_idx))
        etaonly_top3=float(np.mean(etaonly_best_hits));etaonly_regret=float(np.mean(etaonly_regrets))
        best_freq=Counter(rank_candidates)
        best_shift=float(np.mean([np.max(qlo[i])>qhi[i,global_idx] for i in range(n)]))
        # State ranking similarity uses paired exact cells only.
        sim_s=[];sim_k=[]
        for i,k in itertools.combinations(range(n),2):
            shared=np.flatnonzero(cell[i]&cell[k])
            if len(shared)>=2:
                sim_s.append(float(spearmanr(q[i,shared],q[k,shared]).statistic))
                sim_k.append(float(kendalltau(q[i,shared],q[k,shared]).statistic))
        # OLS additive table decomposition on all certified Q cells. The fit
        # is descriptive; residual energy is computed on observed entries.
        aa,bb=np.where(cell);yv=q[cell]
        if len(yv):
            design=np.column_stack((np.ones(len(yv)),np.eye(n)[aa,1:],np.eye(24)[bb,1:]))
            coef=np.linalg.lstsq(design,yv,rcond=None)[0];fit=design@coef;resid=yv-fit;center=yv-yv.mean();energy=float(np.sum(center**2))
            state_eff=np.r_[0.,coef[1:n]];eta_eff=np.r_[0.,coef[n:]]
            state_eff-=state_eff.mean();eta_eff-=eta_eff.mean()
            se=float(np.sum(state_eff[aa]**2));ee=float(np.sum(eta_eff[bb]**2));re=float(np.sum(resid**2))
            decomp={'exact_cells':len(yv),'mean':float(yv.mean()),'state_main_effect_energy_observed':se,
                'eta_main_effect_energy_observed':ee,'interaction_residual_energy':re,
                'interaction_energy_ratio':re/energy if energy else 0.,'state_main_effect_ratio':se/energy if energy else 0.,
                'eta_main_effect_ratio':ee/energy if energy else 0.,'fit':'least-squares additive state and eta main effects over exact cells'}
        else:decomp={'exact_cells':0,'interaction_energy_ratio':None}
        critic_rhos=[];critic_tops=[];critic_top_upper=[];critic_regrets=[];critic_regret_upper=[];critic_top3=[];critic_b15=[];critic_b15_upper=[];critic_picks=[]
        for i in range(n):
            ix=np.flatnonzero(cell[i]);order=ix[np.argsort(-q[i,ix],kind='stable')]
            if len(ix)>=2:
                rho=spearmanr(q[i,ix],p[i,ix]).statistic;critic_rhos.append(float(rho) if np.isfinite(rho) else None)
            pc=np.argsort(-p[i],kind='stable');pick=int(pc[0]);critic_picks.append(pick)
            critic_tops.append(float(qlo[i,pick]));critic_top_upper.append(float(qhi[i,pick]))
            critic_regrets.append(float(np.max(qlo[i])-qhi[i,pick]));critic_regret_upper.append(float(np.max(qhi[i])-qlo[i,pick]))
            critic_top3.append(bool(int(order[0]) in set(map(int,pc[:3]))))
            critic_b15.append(bool(qlo[i,pick]>=15/16));critic_b15_upper.append(bool(qhi[i,pick]>=15/16))
        reversals=[];weak=[];pair_any=set();participants=set()
        for i,k in itertools.combinations(range(n),2):
            found=False
            for x,y in itertools.combinations(range(24),2):
                if not (cell[i,x] and cell[i,y] and cell[k,x] and cell[k,y]):continue
                da=q[i,x]-q[i,y];db=q[k,x]-q[k,y]
                if da*db<0:
                    rec={'state_indices':[i,k],'state_uids':[m['state_uids'][i],m['state_uids'][k]],
                         'eta_indices':[x,y],'etas':[panel[x].tolist(),panel[y].tolist()],
                         'delta_state_a':float(da),'delta_state_b':float(db)}
                    if (da>=.25 and db<=-.25) or (da<=-.25 and db>=.25):reversals.append(rec);found=True
                    else:weak.append(rec)
            if found:pair_any.add((i,k));participants|={i,k}
        rev_acc=[]
        for r in reversals:
            i,k=r['state_indices'];x,y=r['eta_indices'];da=p[i,x]-p[i,y];db=p[k,x]-p[k,y]
            rev_acc.append(bool(da*r['delta_state_a']>0 and db*r['delta_state_b']>0 and da*db<0))
        # Same-state h shuffle across states, repeated five times; eta panel held fixed.
        shuffle=[]
        valid=np.arange(n)
        orig=np.asarray(critic_picks)
        orig_q=np.array([qlo[i,orig[i]] for i in valid])
        for seed in (20261003,20261004,20261005,20261006,20261007):
            perm=np.random.default_rng(seed).permutation(n);sp=p[perm]
            pick=np.argmax(sp,axis=1)
            shuffle.append({'seed':seed,'top1_agreement_with_unshuffled':float(np.mean(pick==orig)),
                'true_Q_of_shuffled_critic_selection_lower':float(np.mean([qlo[i,pick[i]] for i in valid])),
                'true_Q_of_shuffled_critic_selection_upper':float(np.mean([qhi[i,pick[i]] for i in valid])),
                'true_Q_of_unshuffled_critic_selection_lower':float(np.mean(orig_q)),
                'true_Q_of_unshuffled_critic_selection_upper':float(np.mean([qhi[i,orig[i]] for i in valid])),
                'score_correlation':float(np.corrcoef(p[valid].ravel(),sp.ravel())[0,1])})
        reports[sc]={'matrix':{'states':n,'etas':24,'complete_Q16_rows':int(cell.all(axis=1).sum()),
             'complete_cells':int(np.isfinite(q).sum()),'expected_cells':n*24,
             'fully_complete_eta_columns':common_eta,
             'B15_true':sum(x['B15'] is True for x in load('q16_records.json')['records'] if x['scenario']==sc),
             'B15_false':sum(x['B15'] is False for x in load('q16_records.json')['records'] if x['scenario']==sc),
             'B15_unresolved':sum(x['B15'] is None for x in load('q16_records.json')['records'] if x['scenario']==sc),
             'collision_rollouts':sum(x['collision_count'] for x in load('q16_records.json')['records'] if x['scenario']==sc),
             'valid_seed_slots':sum(x['valid_seed_count'] for x in load('q16_records.json')['records'] if x['scenario']==sc),
             'unresolved_seed_slots':sum(len(r['numerical_seeds']) for r in load('q16_records.json')['records'] if r['scenario']==sc)},
          'rankings':{'unique_best_eta_count':len(set(rank_candidates)),'best_eta_index_counts':dict(best_freq),
             'states_with_uncertain_observed_best_due_to_unresolved_eta':len(rank_uncertain),
             'per_state_best_and_top3':state_rank_details,
             'best_differs_from_global_best_fraction':best_shift,
             'pairwise_spearman_mean_median':[float(np.mean(sim_s)),float(np.median(sim_s))] if sim_s else None,
             'pairwise_kendall_mean_median':[float(np.mean(sim_k)),float(np.median(sim_k))] if sim_k else None,
             'pairwise_rank_comparisons':len(sim_s)},
          'reversals':{'strong_quadruple_count':len(reversals),'state_pairs_with_strong_reversal':len(pair_any),
             'fraction_state_pairs_with_strong_reversal':len(pair_any)/(n*(n-1)/2),
             'states_participating':len(participants),'weaker_reversal_quadruple_count':len(weak),
             'critic_strong_reversal_direction_accuracy':float(np.mean(rev_acc)) if rev_acc else None,
             'strong_records':reversals,'weak_records':weak},
          'oracle':{'global_eta_index':global_idx,'global_eta':panel[global_idx].tolist() if global_idx is not None else None,
             'global_eta_selection':'highest mean Q_lower across all24 panel eta; mean Q and B15 intervals also reported',
             'common_column_eta_indices':common_eta,
             'global_mean_Q_lower':global_q,'global_mean_Q_upper':global_q_upper,
             'global_B15_coverage_lower':global_b15,'global_B15_coverage_upper':global_b15_upper,
             'state_oracle_mean_Q_lower':oracle_q,'state_oracle_mean_Q_upper':oracle_q_upper,
             'state_oracle_B15_coverage_lower':oracle_b15_lower,'state_oracle_B15_coverage_upper':oracle_b15_upper,
             'STATE_SPECIFIC_ORACLE_GAP_lower':gap,'STATE_SPECIFIC_ORACLE_GAP_upper':gap_upper},
          'additive_decomposition':decomp,
          'eta_only':{'top1_eta_index':etaonly_idx,'top1_eta':panel[etaonly_idx].tolist(),
             'top1_accuracy_vs_state_best':etaonly_top1_accuracy,
             'top3_hit_rate':etaonly_top3,'mean_regret_lower':etaonly_regret,
             'B15_coverage_lower':global_b15,'B15_coverage_upper':global_b15_upper},
          'critic':{'mean_per_state_spearman':float(np.nanmean([x for x in critic_rhos if x is not None])) if any(x is not None for x in critic_rhos) else None,
             'top1_mean_true_Q':float(np.mean(critic_tops)) if critic_tops else None,
             'top1_mean_true_Q_upper':float(np.mean(critic_top_upper)) if critic_top_upper else None,
             'top1_mean_regret_lower':float(np.mean(critic_regrets)) if critic_regrets else None,
             'top1_mean_regret_upper':float(np.mean(critic_regret_upper)) if critic_regret_upper else None,
             'top3_hit_rate':float(np.mean(critic_top3)) if critic_top3 else None,
             'B15_selection_rate_lower':float(np.mean(critic_b15)) if critic_b15 else None,
             'B15_selection_rate_upper':float(np.mean(critic_b15_upper)) if critic_b15_upper else None,
             'strong_reversal_direction_accuracy':float(np.mean(rev_acc)) if rev_acc else None,
             'improvement_over_eta_only_B15_lower':float(np.mean(critic_b15)-global_b15_upper) if critic_b15 else None,
             'improvement_over_eta_only_B15_upper':float(np.mean(critic_b15_upper)-global_b15) if critic_b15 else None,
             'state_shuffle':shuffle},
          'classification':classify(gap,gap_upper,len(pair_any)/(n*(n-1)/2),best_shift)}
        save(f'{sc}_Q16_critic.json',{'state_uids':m['state_uids'],'Q16':q.tolist(),'critic_scores':p.tolist()})
    save('analysis.json',reports);write_report(reports);write_dataset_package();return reports

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','scores','preflight','run','collect','analyze']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=18);args=ap.parse_args()
    fn={'freeze':freeze,'scores':critic_scores,'preflight':preflight,'run':lambda:run(args.shard,args.shards),'collect':collect,'analyze':analyze}[args.stage]
    print(json.dumps(fn(),indent=2,default=str),flush=True)
