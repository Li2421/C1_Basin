"""Read saved predictions and official code; do not rerun or rewrite records."""
from experiment import *
import numpy as np

def main():
    rows=[]
    for stage in ('paper_primary','paper_secondary'):
        for path in (HERE/stage/'raw').glob('*.json'):
            r=read(path)
            if r['case']['state']['scene'] not in ('lit_doorway','lit_intersection_secondary'):continue
            with np.load(r['trace_path']) as z:
                displacement=np.linalg.norm(z['positions'][1]-z['positions'][0],axis=-1).tolist();epochs=len(z['positions'])-1
                assert z['optimizer_states'].shape[1]==6
            rows.append(dict(raw_path=str(path),raw_sha256=sha(path),rollout_uid=r['rollout_uid'],legacy_reported_seconds=r['physical_time_seconds'],legacy_episode_length=r['episode_length'],corrected_seconds=.6*epochs,actual_simulator_steps_per_agent=6*epochs,epoch_spacing_seconds=.6,first_epoch_displacement_m=displacement,max_displacement_if_only_three_steps_m=.09,execution_trace_unchanged=True))
    assert len(rows)==4
    source=HERE/'external/mpc-cbf/mpc_cbf.py'
    write(HERE/'TIME_METADATA_CORRECTION.json',dict(utc=utc(),records=rows,source=str(source),source_sha256=sha(source),reason='Simulator.x0 initialized only in constructor lines383-385; run_simulation lines403-407 advances3steps; run_simulation_to_get_final_condition only resets MPC observation x1 at411, then advances the same simulator3more steps at425-436. No simulator reset between calls. Returned poses are after6steps per epoch.',new_rollouts=0,old_raw_and_DB_rows_untouched=True))
    seal('TIME_METADATA_CORRECTION.receipt.json',[HERE/'TIME_METADATA_CORRECTION.json',HERE/'correct_time_metadata.py'],analysis_only=True)
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n- 门口/文献交叉口物理时间勘误：两次3步调用复用同一模拟器且不重置，实际每轮6步，50轮30秒。早期raw中的15秒与150步是错误元数据；首轮位移约.18米大于3步速度界.09米、优化器保存6条状态，均支持6步。执行轨迹正确，全部保留；不重跑、不UPDATE共享DB。报告从旧预测按.6秒epoch重算窗口与时间，见TIME_METADATA_CORRECTION收据。\n')

if __name__=='__main__':main()
