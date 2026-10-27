"""Compute the native fixed-menu crossfit reference on completed evidence only."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from completed_diagnosis_functions import collect_completed_diagnoses
import importlib.util
import fcntl
import time
import gc


def main():
    with (HERE/'COMPLETED_FIXED_CROSSFIT.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('COMPLETED_FIXED_CROSSFIT_CODE.receipt.json')
        verify_receipt('COMPLETED_DIAGNOSIS_NUMBERS.receipt.json')
        verify_receipt('LOCAL_DIAGNOSIS_SCOPE.receipt.json')
        os.nice(10)
        write(HERE/'COMPLETED_FIXED_CROSSFIT_PID.json',dict(utc=utc(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
        spec=importlib.util.spec_from_file_location('sealed_v2_fixed_crossfit_statistics',HERE/'statistics.py')
        stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
        records,sources=collect_completed_diagnoses(read(HERE/'LOCAL_DIAGNOSIS_SCOPE.json')['originals'])
        assert len(records)==200640
        source_digest=hashlib.sha256(canonical(sources).encode()).hexdigest()
        del sources
        results={};rows=[];labels={'central':'集中式','distributed':'分布式'}
        for scene in NAMES:
            keys={(r['N'],r['state']) for r in records if r['scene']==scene and r['layer']=='distributed' and r['method']=='learned_scene'}
            # Exactly the main program's evaluation scope. Circle's main and
            # forthcoming learned-evaluation state sets are both the same16.
            subset=[r for r in records if r['scene']==scene and (not keys or (r['N'],r['state']) in keys)]
            expected_states=16 if scene=='lit_circle_five' else 32
            assert len({(r['N'],r['state']) for r in subset})==expected_states
            results[scene]={}
            for layer in ('central','distributed'):
                result=stats.crossfit_oracle(subset,layer,scene)
                results[scene][layer]=result
                table=result['result']
                if table['status']=='complete':
                    points={r['method']:r for r in table['statistics'] if r['type']=='type_equal'}
                    rows.append([NAMES[scene],labels[layer],expected_states,*[f"{points[m]['success_rate']*100:.4f}%" for m in (*stats.FIXED,'fixed_oracle_crossfit')]])
                write(HERE/'COMPLETED_FIXED_CROSSFIT_PROGRESS.json',dict(utc=utc(),scene=scene,layer=layer,completed_summaries=sum(len(v) for v in results.values()),new_rollouts=0,new_training=0))
        write(HERE/'COMPLETED_FIXED_CROSSFIT.json',dict(utc=utc(),evidence='DEVELOPMENT',results=results,full_main_complete=False,new_rollouts=0,new_training=0,bootstrap_iterations=5000,logical_input_records=len(records),source_digest_sha256=source_digest,scope='Same frozen statistics.crossfit_oracle and state scopes as eventual main; descriptive controller-specific fixed-menu crossfit, not the learned policy'))
        header=['场景','安全层','评估state','只加安全','固定η10','固定η13','人定切换','固定菜单state交叉验证对照']
        text='已完成证据上的固定菜单 state 交叉验证对照（开发证据，非主线完成标志）。\n\n'
        text+='按每个 controller 的排序 (N,index) 奇偶划分 state，两 seed 同折；只在另一折选择固定η10、固定η13、人定切换中成功下界最高者，再在当前折评估。固定菜单平局顺序沿用原程序。这是描述性 controller 内对照，不是新部署方法，不是留一类型的学习版，也不是逐局看答案选动作。四场景的范围与最终主程序一致：圆形围场16组，其余32组学习评估子集。\n\n'
        text+='| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'
        text+=''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
        text+='\n类型→controller→state等权，两个seed配对；unknown按下界计。5000次配对bootstrap、unknown上界、差值区间与选择细节见同名JSON。只读取已有raw和诊断，没有新rollout或训练。圆形围场学习评估、第二阶段及结束审计仍在后续流程中。最终MIGRATION_RESULTS生成后会核对对应结果完全一致。\n'
        (HERE/'COMPLETED_FIXED_CROSSFIT.md').write_text(text)
        seal('COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json',[HERE/n for n in ('COMPLETED_FIXED_CROSSFIT.json','COMPLETED_FIXED_CROSSFIT.md','COMPLETED_FIXED_CROSSFIT_CODE.receipt.json')],full_main_complete=False,new_rollouts=0,new_training=0)
        del records;gc.collect()
        print('Completed-scope crossfit numerical reference sealed; main workers continue',flush=True)
        while not (HERE/'MIGRATION_NUMBERS.receipt.json').exists():time.sleep(30)
        verify_receipt('MIGRATION_NUMBERS.receipt.json')
        final=read(HERE/'MIGRATION_RESULTS.json')['results']
        for scene,by_layer in results.items():
            for layer,result in by_layer.items():assert result==final[scene]['crossfit_fixed_oracle'][layer],(scene,layer)
        write(HERE/'COMPLETED_FIXED_CROSSFIT_COMPARISON.json',dict(utc=utc(),passed=True,identical_scopes=8,new_rollouts=0))
        seal('COMPLETED_FIXED_CROSSFIT_COMPARISON.receipt.json',[HERE/n for n in ('COMPLETED_FIXED_CROSSFIT_COMPARISON.json','COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json','MIGRATION_NUMBERS.receipt.json')],passed=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'COMPLETED_FIXED_CROSSFIT_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc()))
        raise
