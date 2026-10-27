"""Verify completed saved diagnoses and report candidates without new simulations."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from completed_diagnosis_functions import collect_completed_diagnoses, weighted_rates
from result_inventory import qualification
import importlib.util
import collections
import fcntl
import csv
import time


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'


def main():
    with (HERE/'COMPLETED_DIAGNOSIS_AUDIT.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('COMPLETED_DIAGNOSIS_ANALYSIS_CODE.receipt.json')
        verify_receipt('LOCAL_DIAGNOSIS_SCOPE.receipt.json')
        verify_receipt('COMPLETED_SCENE_NUMBERS.receipt.json')
        exit_record=read(HERE/'LOCAL_SAVED_DIAGNOSIS_EXIT.json')
        assert exit_record['reason']=='All prepared completed-source diagnoses available'
        assert exit_record['new_rollouts']==0
        os.nice(10)
        write(HERE/'COMPLETED_DIAGNOSIS_AUDIT_PID.json',dict(utc=utc(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
        scope=read(HERE/'LOCAL_DIAGNOSIS_SCOPE.json');verified={};coverage=[]
        started=time.monotonic()
        for stage in scope['stages']:
            verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
            digest=hashlib.sha256();count=0
            for entry in read(HERE/stage/'GROUP_INDEX.json'):
                assert sha(entry['file'])==entry['sha256'],entry['file']
                for row in read(entry['file']):
                    rid=row['rollout_uid'];path=HERE/'DIAGNOSTICS'/f'{rid}.json'
                    if rid not in verified:
                        assert sha(row['path'])==row['sha256'],row['path']
                        raw=read(row['path']);diag=read(path)
                        assert raw['rollout_uid']==diag['rollout_uid']==rid
                        assert diag['trace_sha256']==raw['trace_sha256'],rid
                        assert sha(diag['source']['path'])==diag['source']['sha256'],rid
                        assert read(diag['source']['path'])['trace_sha256']==diag['trace_sha256'],rid
                        assert len(diag['local'])==raw['case']['state']['N'],rid
                        assert len(diag.get('grid',[]))==(0 if raw['numerical_failure'] else 9),rid
                        if raw['numerical_failure']:assert set(diag['local'])=={'unknown'},rid
                        verified[rid]=dict(rollout_uid=rid,raw_sha256=row['sha256'],diagnostic_sha256=sha(path),trace_sha256=raw['trace_sha256'])
                    else:
                        assert verified[rid]['raw_sha256']==row['sha256'],rid
                    digest.update(canonical(verified[rid]).encode());count+=1
            coverage.append(dict(stage=stage,verified_logical_diagnoses=count,ordered_evidence_sha256=digest.hexdigest()))
            write(HERE/'COMPLETED_DIAGNOSIS_AUDIT_PROGRESS.json',dict(utc=utc(),stage=stage,verified_unique=len(verified),verified_logical=sum(r['verified_logical_diagnoses'] for r in coverage),new_rollouts=0))
        total=sum(r['verified_logical_diagnoses'] for r in coverage)
        assert total==exit_record['written']+exit_record['already_cached']
        records,sources=collect_completed_diagnoses(scope['originals'])
        assert len(records)==total
        assert {r['rollout_uid'] for r in records}==set(verified)
        spec=importlib.util.spec_from_file_location('sealed_v2_completed_diag_statistics',HERE/'statistics.py')
        stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
        census=stats.controller_census(records)
        qualified={s:{r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES}
        valid=[r for r in records if r['controller'] in qualified[r['scene']]]
        failure_keys={(r['scene'],r['layer'],r['controller']) for r in census if r['controller_failure_definite']}
        rates={s:weighted_rates([r for r in valid if r['scene']==s]) for s in NAMES}
        removed={s:weighted_rates([r for r in valid if r['scene']==s and (r['scene'],r['layer'],r['controller']) not in failure_keys]) for s in NAMES}
        discrimination=[]
        for scene in NAMES:
            for layer in ('central','distributed'):
                rr=[r for r in census if r['scene']==scene and r['layer']==layer and r['qualified']]
                discrimination.append(dict(scene=scene,layer=layer,qualified=len(rr),controller_failure_definite=sum(r['controller_failure_definite'] for r in rr),controller_failure_possible=sum(r['controller_failure_possible'] for r in rr),state_related_failure_definite=sum(r['state_related_failure_definite'] for r in rr),state_related_failure_possible=sum(r['state_related_failure_possible'] for r in rr),controllers_with_4a_prime_candidate=sum(r['at_least_one_4a_prime'] for r in rr),eligible_controllers_with_4a_prime_candidate=sum(r['at_least_one_4a_prime'] and r['deadlock_evidence_eligible'] for r in rr)))
        # The already sealed outcome snapshot must agree on every complete fixed scope.
        previous=read(HERE/'COMPLETED_SCENE_RESULTS.json')['results']
        for scene in NAMES:
            for row in previous[scene]['fixed_full']['statistics']:
                if row['type']!='type_equal':continue
                actual=rates[scene][row['layer']+'|'+row['method']]['rates']
                assert abs(actual['success']-row['success_rate'])<1e-14,(scene,row)
                assert abs(actual['upper']-row['success_upper'])<1e-14,(scene,row)
        result=dict(utc=utc(),evidence='DEVELOPMENT',full_main_complete=False,new_rollouts=0,new_training=0,
            input_scope=scope['originals'],diagnosis_coverage=coverage,logical_diagnoses=total,unique_diagnoses=len(verified),
            diagnostic_reference='Frozen failure_analysis.py, primary W=5 seconds/tau=.05; timeout is not deadlock',
            weighted_rates=rates,per_layer_excluding_definite_controller_failures=removed,discrimination=discrimination,
            controller_census=census,elapsed_seconds=time.monotonic()-started,
            trace_verification_scope='Raw/diagnostic hashes and stored trace hashes cross-checked; trace bytes were verified by the diagnosis producer. This is not the final source integrity audit.',
            pending='Circular learned labels, model and evaluation; second stage and final audits remain pending')
        write(HERE/'COMPLETED_DIAGNOSIS_RESULTS.json',result)
        with (HERE/'COMPLETED_DIAGNOSIS_CONTROLLERS.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(census[0]));writer.writeheader();writer.writerows(census)
        labels={'central':'集中式','distributed':'分布式'}
        headers=['场景','安全层','合格实例','明确 controller 级失败','明确状态相关失败','有4a′候选的实例','排除明确实例失败后有候选']
        rows=[[NAMES[r['scene']],labels[r['layer']],r['qualified'],r['controller_failure_definite'],r['state_related_failure_definite'],r['controllers_with_4a_prime_candidate'],r['eligible_controllers_with_4a_prime_candidate']] for r in discrimination]
        text='现有完整批次的逐机器人诊断（开发证据）。第一阶段尚未完成，圆形围场学习评估、第二阶段和结束审计仍在后续流程中。\n\n'
        text+=f'已核验 {total:,} 个逻辑诊断引用、{len(verified):,} 个不同 rollout UID；没有新 rollout 或训练。原始记录、诊断及其已存轨迹哈希相互核对，轨迹文件内容由生成诊断时的原程序核验。本次核验不替代最终完整性审计。\n\n'
        text+=table(headers,rows)+'\n'
        rate_rows=[]
        for scene in NAMES:
            for layer in labels:
                r=rates[scene][layer+'|safety']['rates']
                remaining=removed[scene].get(layer+'|safety')
                rate_rows.append([NAMES[scene],labels[layer],*[f"{100*r[k]:.4f}%" for k in ('success','collision','timeout','unknown','deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode')],f"{100*remaining['rates']['deadlock_episode']:.4f}%" if remaining else 'N/A'])
        text+=table(['场景','安全层','成功','碰撞','超时','unknown','逐机器人4a′候选','墙前4a″','原联合锥参照','实际乘积联合锥','排除明确实例失败后4a′候选'],rate_rows)
        text+='\n率按类型→controller→state→配对 seed 等权。总体率保留全部合格实例；成功率≤5%的明确 controller 级失败不作死锁证据。4a′是有限窗口操作性候选，超时本身不是死锁。圆形围场速度较小，静止窗口较弱，单个候选不能证明永久死锁。排除实例后的各层范围可能不同，严格跨层配对推断以后续主结果为准。完整区间、实例分类和各做法结果见同名 JSON 与 CONTROLLERS.csv；成功率的5000次 bootstrap区间仍见 COMPLETED_SCENE_RESULTS.md/.json。\n'
        (HERE/'COMPLETED_DIAGNOSIS_RESULTS.md').write_text(text)
        seal('COMPLETED_DIAGNOSIS_NUMBERS.receipt.json',[HERE/n for n in ('COMPLETED_DIAGNOSIS_RESULTS.json','COMPLETED_DIAGNOSIS_RESULTS.md','COMPLETED_DIAGNOSIS_CONTROLLERS.csv','COMPLETED_DIAGNOSIS_ANALYSIS_CODE.receipt.json','LOCAL_SAVED_DIAGNOSIS_EXIT.json')],new_rollouts=0,new_training=0,full_main_complete=False)
        print(f'Verified {total} logical / {len(verified)} unique completed diagnoses; main continues',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'COMPLETED_DIAGNOSIS_AUDIT_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc()))
        raise
