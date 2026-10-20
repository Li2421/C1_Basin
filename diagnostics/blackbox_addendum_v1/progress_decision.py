from common import *
from progress import ProgressSwitch,offline_switch
from audit_utils import source_inventory
import struct

def main():
    prereg();assert (HERE/'A_COMPLETE.receipt.json').exists(),'Part A must finish first';assert not (HERE/'PSTAR_DECISION.receipt.json').exists();source=R35/'PSTAR_CALIBRATION_CANDIDATES.json';x=read(source);selected=[r for r in x['source_index'] if r['coverage']=='observed500'];assert len(selected)==699;value=float(np.median([r['p500_candidate'] for r in selected]));expected=.9681620884830543
    if struct.pack('=d',value)!=struct.pack('=d',expected):write(HERE/'B_GAP.json',dict(utc=utc(),reason='Independent observed500 median differs bitwise',computed=value,expected=expected));return
    text='用户决定：候选1，只用确实跑到第500步的699条，p* = 0.9681620884830543。第500步之前提前成功的、数值失败的都没有第500步读数，一律不用。'
    write(HERE/'PSTAR_CALIBRATION_RECOMPUTED.json',dict(utc=utc(),selected=[dict(rollout_uid=r['rollout_uid'],p500_candidate=r['p500_candidate']) for r in selected],median=value,median_float64_hex=np.asarray(value,np.float64).tobytes().hex(),source_sha256=sha(source),excluded_early_success=195,excluded_numerical_unknown=2));(HERE/'PSTAR_DECISION.md').write_text(text+'\n\n独立重算仅使用 coverage=observed500 的699条。排除了最快完成的episode，因此门槛比终止保持定义0.9728120313305867略低，切换略早。既有兼容实验已经执行本规则，本轮不重跑；其事前决定收据随来源保留。\n')
    seal('PSTAR_DECISION.receipt.json',[HERE/'PSTAR_DECISION.md',HERE/'PSTAR_CALIBRATION_RECOMPUTED.json',source],original_user_decision=text,pstar=value,observed500_count=699,float64_bitwise_equal=True,before_any_new_progress_schedule_rollout=True,new_progress_rollouts_this_round=0,historical_source_decision=str(D/'axis4a_addendum_v1/PSTAR_DECISION.receipt.json'))
    checks=[]
    for H in [500,700,1000]:
        goals=np.ones((4,2))*10;p0=np.zeros((4,2));path=np.asarray([p0+(goals-p0)*min(t/H,1) for t in range(H+1)]);rule=ProgressSwitch(path[0],goals,H,value)
        for t in range(H):rule.eta_index(t,path[t],goals)
        expected=offline_switch(path,goals,H,value);assert rule.switched_at==expected;checks.append(dict(H=H,online=rule.switched_at,offline=expected))
    # Archived observed traces: no virtual action at a terminal snapshot.
    for row in selected[:16]:
        assert sha(row['trace_path'])==row['trace_sha256'];z=np.load(row['trace_path']);path=z['positions'];goals=z['goals'];rule=ProgressSwitch(path[0],goals,700,value)
        for t in range(min(700,len(path)-1)):rule.eta_index(t,path[t],goals)
        assert rule.switched_at==offline_switch(path,goals,700,value)
    write(HERE/'SELECTED_PROGRESS_TEST.json',dict(utc=utc(),passed=True,synthetic=checks,archived_traces=16,new_physical_steps=0));source_inventory('imports_progress_decision.json')
if __name__=='__main__':main()
