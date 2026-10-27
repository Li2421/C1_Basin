"""Evaluate official initialization matrices; zero simulated/control steps."""
from experiment import *
import numpy as np

def main():
    sys.path.insert(0,str(HERE/'external/IMPC-DR/2D'));import SET
    from uav import uav
    Ac=np.block([[np.zeros((2,2)),np.eye(2)],[np.zeros((2,4))]]);Bc=np.r_[np.zeros((2,2)),np.eye(2)];rows=[]
    for h,K in ((.2,10),(.15,12)):
        SET.initialize_set(1,np.zeros((1,2)),np.zeros((1,2)),np.ones((1,2)),.3,.1,h,K,1)
        obj=uav(0,np.zeros(2),np.zeros(2),np.ones(2),K);A=obj.VA[:4];B=obj.VB[:4,:2]
        exact_A=np.eye(4)+h*Ac;code_B=np.r_[np.eye(2)*h**3/(2*(1-h/2)**2),np.eye(2)*h/(1-h/2)]
        np.testing.assert_allclose(A,exact_A,rtol=0,atol=1e-15);np.testing.assert_allclose(B,code_B,rtol=1e-14,atol=1e-15)
        correct_B=np.linalg.solve(np.eye(4)-h/2*Ac,h*Bc)
        rows.append(dict(h=h,K=K,official_A=A,official_B=B,paper_euler_A=np.eye(4)+h*Ac,paper_euler_B=h*Bc,correct_tustin_B=correct_B,reason='uav.py:295 overwrites continuous A with discrete A; line296 reuses the overwritten A when computing B',new_control_steps=0))
    p=HERE/'OFFICIAL_DYNAMICS_AUDIT.json';write(p,dict(utc=utc(),source_sha256=sha(HERE/'external/IMPC-DR/2D/uav.py'),records=rows,source_preserved=True,new_rollouts=0))
    seal('OFFICIAL_DYNAMICS_AUDIT.receipt.json',[p,HERE/'audit_official_dynamics.py'],initialization_only=True)
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n- 官方IMPC动力学复核勘误：get_dynamic先用离散A覆盖连续A，再在B表达式重复使用该A，因此实际B不是正确Tustin/ZOH。h=.2时位置输入系数.0049382716、速度输入系数.2222222222；h=.15时按同一公式。OFFICIAL_DYNAMICS_AUDIT直接读官方初始化矩阵验证，未执行控制/rollout。此前参数说明中的“.5h²位置块”不正确，报告修正；官方代码和全部既有轨迹原样保留，不重跑。\n')

if __name__=='__main__':main()
