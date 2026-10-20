"""Row-by-row historical action compatibility, without reading parallel directories."""
from common import *
from shared_rollout_db.src.rollout_db import connect,canonical,eta_identity
from collections import Counter

def audit():
    exact=read(HERE/'ETA_EXACT.json');etauids={eta_identity(v['values'])[0]:k for k,v in exact.items()};accepted=[];excluded=[]
    with connect(True) as db:
        rows=db.execute("SELECT r.*,c.config_json,c.scenario_uid,s.name scene,t.physical_state_json,t.goals_geometry_json,t.provenance_json FROM rollout r JOIN controller_config c USING(controller_uid) JOIN scenario s ON s.scenario_uid=c.scenario_uid JOIN state t USING(state_uid) WHERE r.experiment_uid NOT LIKE '%teacher_student%' AND r.experiment_uid NOT LIKE '%decentralized_blackbox_planners%' AND (s.name LIKE '%ring%' OR s.name LIKE '%Ring%' OR r.experiment_uid='exp_axis3_cross_scene_v1')").fetchall()
    for row in rows:
        r=dict(row);cfg=json.loads(r.pop('config_json'));scene=r['scene'];reason=None;arm=None
        text=canonical(cfg).lower()
        if 'teacher_student_macflow_v1' in text or 'decentralized_blackbox_planners_v1' in text or 'round30' in text:reason='Forbidden parallel provenance: do not access source directories'
        elif r['conflict_quarantined']:reason='Quarantined historical evidence'
        elif r['compatibility_quality']!='EXACT_REUSE':reason='Insufficient exact execution identity'
        elif str(cfg.get('chain','')).upper()=='FF' or 'nominal_ff' in text or 'affine_ff' in text:reason='Flow-internal correction placement differs from post-projection correction menu'
        elif str(cfg.get('chain','')).upper()!='TT' and 'project_correct_reproject' not in text and 'tt_project' not in text and 'original_tt' not in text:reason='Post-projection correction identity not explicit'
        elif any(k in text for k in ['centralized_sampling','prioritized_astar','cross_entropy','"cem"']):reason='Not one of the requested historical controller classes'
        elif 'continuing_right_hand' in cfg or 'rhr_rule' in cfg or 'right_hand' in str(cfg.get('execution_semantics','')).lower():reason='Historical center-based clockwise right-hand direction differs from goal-relative direction'
        else:
            schedule=cfg.get('eta_schedule',cfg.get('schedule'))
            if 'progress_switch' in cfg or 'progress_threshold' in cfg:
                prog=cfg.get('progress_switch',{});threshold=prog.get('threshold',cfg.get('progress_threshold')) if isinstance(prog,dict) else cfg.get('progress_threshold')
                if threshold==.9681620884830543:arm='Sprogress'
                else:reason='Progress-switch identity not exact'
            elif isinstance(schedule,dict):
                early=schedule.get('eta_a',schedule.get('early'));late=schedule.get('eta_b',schedule.get('late'));switch=schedule.get('switch_step',schedule.get('t'));H=cfg.get('horizon',cfg.get('max_steps'))
                try:
                    a=eta_identity(early)[0];b=eta_identity(late)[0]
                    if b==eta_identity(exact['10']['values'])[0] and int(switch)==round(5*int(H)/7) and a in etauids and etauids[a] in ['13','15']:arm='S'+etauids[a]
                    else:reason='Switch schedule differs from5/7 per-state horizon'
                except Exception:reason='Schedule identity unrecoverable'
            elif r['eta_uid'] in etauids:arm='F'+etauids[r['eta_uid']]
            else:reason='Exact eta is outside the correction menu'
        if reason:excluded.append(dict(rollout_uid=r['rollout_uid'],experiment_uid=r['experiment_uid'],scene=scene,reason=reason));continue
        try:
            physical=json.loads(r['physical_state_json']);assert all(k in physical for k in ['positions','velocities','goals'])
        except Exception:
            excluded.append(dict(rollout_uid=r['rollout_uid'],experiment_uid=r['experiment_uid'],scene=scene,reason='Cannot recover exact physical triple'));continue
        r.update(arm=arm,profile=cfg,physical=physical)
        accepted.append(r)
    write(HERE/'HISTORICAL_COMPATIBLE_LABELS.json',accepted)
    write(HERE/'HISTORICAL_LABEL_EXCLUSIONS.json',excluded)
    summary=dict(accepted_rows=len(accepted),accepted_by_arm=dict(Counter(r['arm'] for r in accepted)),accepted_by_scene=dict(Counter(r['scene'] for r in accepted)),excluded_by_reason=dict(Counter(r['reason'] for r in excluded)),unknown_retained=sum(r['numerical_failure'] for r in accepted),existing_scene_new_rollouts=0,source_G_labels=0)
    write(HERE/'HISTORICAL_ACTION_COMPATIBILITY.json',summary)
    seal('HISTORICAL_LABELS_FROZEN.receipt.json',[HERE/'HISTORICAL_COMPATIBLE_LABELS.json',HERE/'HISTORICAL_LABEL_EXCLUSIONS.json',HERE/'HISTORICAL_ACTION_COMPATIBILITY.json',HERE/'history_labels.py'],before_new_target_labels=True)
    print(summary,flush=True)

if __name__=='__main__':audit()
