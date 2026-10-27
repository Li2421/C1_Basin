"""Status after user-directed cancellation; never schedules or recovers work."""
from experiment import *
from closure_scope import FLOW_NAMES,STOP_REASON,circle_gap
import subprocess


def main():
    rows=[]
    for p in HERE.glob('*_CASES.json'):
        if p.name.endswith('_MISSING_CASES.json'):continue
        stage=p.name[:-len('_CASES.json')];raw={q.stem for q in (HERE/stage/'raw').glob('*.json')}
        planned=len(read(p));todo={c['rollout_uid'] for c in read(HERE/f'{stage}_MISSING_CASES.json')}
        complete=planned-len(todo)+len(todo&raw)
        disposition='用户停止；未落盘丢弃，不记unknown、不恢复' if stage=='safety_lit_circle_five' else '废弃的0步实现分支，不继续执行' if stage=='capability_lit_narrow_passage' else '已完成' if complete==planned else '仅保留原worker继续完成'
        rows.append(dict(stage=stage,planned=planned,logical_completed=complete,
                         raw_retained=len(raw),unwritten=len(todo-raw),disposition=disposition))
    queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-o','%.18i %.30j %.8T %.10M %.6C %.20b']).decode()
    delivered=(HERE/'DELIVERY_SUMMARY.json').exists()
    state=dict(utc=utc(),phase='v1_closed_user_scope' if delivered else 'v1_closing_three_completed_flow_scenes',
               stop_reason=STOP_REASON,coordinator_stopped=True,no_new_v1_submissions=True,
               included_flow_scenes=list(FLOW_NAMES),excluded_circle=circle_gap(),
               learning_pilots='未启动；按用户决定不再提交',stages=rows,squeue=queue,
               original_four_scene_objective_completed=False,
               v1_user_authorized_closure_completed=delivered)
    write(HERE/'STATUS.json',state)
    text=['# v1收尾状态','',state['utc'],'',
          '按用户决定停止圆形流程版并转入v2。协调器已退出；v1不再提交新批次。仅论文版缓冲Voronoi单元修正分支原worker继续到完成。最终流程结论只包含窄通道、二维拥挤随机、门口三个场景，不作四场景合并。', '',
          f"圆形流程版保留{circle_gap()['completed_raw_retained']}条已落盘raw及journal；{circle_gap()['unwritten_discarded']}条未落盘请求丢弃，不是unknown，不恢复。两类行为克隆与扩展未启动。", '',
          'v1当前用户授权收尾已完成。下一阶段执行指定v2文档，v1目录只读。' if delivered else '正在合并已完成journal、封存三个场景统计并等待原论文worker结束，之后做结束审计。','']
    text += [f"- {r['stage']}：逻辑完成{r['logical_completed']}/{r['planned']}；{r['disposition']}。" for r in rows]
    text += ['','```text',queue,'```']
    (HERE/'STATUS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
