"""Recompute exact sampled LQR predictions algebraically, without new rollouts."""
from experiment import *
import numpy as np
from scipy.linalg import solve_continuous_are

def main():
    prereg();assert not (HERE/'CIRCLE_SPEED.receipt.json').exists()
    states=[s for s in read(HERE/'PAPER_STATES.json') if s['scene']=='lit_circle_five'];assert len(states)==100
    A=np.block([[np.zeros((2,2)),np.eye(2)],[np.zeros((2,4))]]);B=np.r_[np.zeros((2,2)),np.eye(2)]
    P=solve_continuous_are(A,B,.2*np.eye(4),np.eye(2));K=B.T@P;dt=.05
    Ad=np.block([[np.eye(2),dt*np.eye(2)],[np.zeros((2,2)),np.eye(2)]]);Bd=np.r_[.5*dt**2*np.eye(2),dt*np.eye(2)]
    M=Ad-Bd@K;powers=np.asarray([np.linalg.matrix_power(M,t) for t in range(2001)])
    initial=np.asarray([np.c_[np.asarray(s['positions'])-np.asarray(s['goals']),np.zeros((5,2))] for s in states])
    prediction=np.einsum('tij,naj->tnai',powers,initial);speed=np.linalg.norm(prediction[...,2:],axis=-1)
    value=float(np.quantile(speed,.95,method='linear'))
    file=HERE/'NOMINAL_LQR_REFERENCE_PREDICTIONS.npz';np.savez_compressed(file,relative_states=prediction,speeds=speed,gain=K,closed_loop_matrix=M,times=np.arange(2001)*dt)
    write(HERE/'CIRCLE_SPEED.json',dict(utc=utc(),nominal_lqr_speed_p95=value,definition='Exact zero-order-held acceleration, sample-feedback LQR; full 100-second nominal reference',quantile_method='linear',source_states_sha256=sha(HERE/'PAPER_STATES.json'),source_paper_code_sha256=sha(HERE/'paper.py'),reference_path=str(file),reference_sha256=sha(file),samples=int(speed.size),states=100,agents_per_state=5,time_samples=2001,formula='x(t)=((Ad-Bd*K)^t)*x0',new_environment_rollouts=0,new_optimization_rollouts=0,continuous_reference_in_old_paper_outputs_not_used=True))
    seal('CIRCLE_SPEED.receipt.json',[HERE/'CIRCLE_SPEED.json',file,HERE/'calibrate_speed.py'],before_circle_flow=True)
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('- 圆形围场速度界校准：paper.py中附带的连续时间LQR参考不代表50ms采样保持控制，未用于速度分位数。按同一已封存100组初始条件、LQR参数及二阶保持动力学，用矩阵幂闭式重算预测；不运行环境、不生成新的优化闭环。速度95分位与全部预测在流程版之前独立封存。\n')
    print(value,speed.size)
if __name__=='__main__':main()
