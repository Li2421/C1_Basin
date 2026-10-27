from experiment import *
import ctypes
import numpy as np
import time
from states import LiteratureConfig
from flow_scene import LiteratureEnv

def main():
    P=ctypes.POINTER(ctypes.c_double);I=ctypes.POINTER(ctypes.c_int);libs=[]
    for name in ('astar_kernel_memo.so','astar_kernel_certified.so'):
        lib=ctypes.CDLL(str(HERE/name));lib.astar.argtypes=[ctypes.c_double]*4+[ctypes.c_int,P,P,ctypes.c_int,I,P,ctypes.c_int,P,ctypes.c_double,P];lib.astar.restype=ctypes.c_int;libs.append(lib)
    state=dict(scene='lit_circle_five',N=1,index=0,positions=[[0.,0.]],goals=[[8.99,0.]],velocities=[[0.,0.]])
    env=LiteratureEnv(state,LiteratureConfig('lit_circle_five',1,2.,.05,.37365109585434214,2000));walls=np.ascontiguousarray(env.walls);lens=np.zeros(0,np.int32);reserved=np.zeros((0,2));rows=[]
    for i in range(16):
        # Both true endpoints have a geometric path inside the arena; the
        # rounded (9,0) cell fails the unchanged 2.005m wall margin.
        start=np.asarray([float(i%4)-3.,float(i//4)-2.],np.float64);goal=np.asarray([8.99,0.],np.float64);outputs=[];elapsed=[]
        for lib in libs:
            out=np.zeros((22,2));t=time.monotonic();length=lib.astar(1.,0.,float('inf'),4.,20,start.ctypes.data_as(P),goal.ctypes.data_as(P),0,lens.ctypes.data_as(I),reserved.ctypes.data_as(P),len(walls),walls.ctypes.data_as(P),2.005,out.ctypes.data_as(P));outputs.append(length);elapsed.append(time.monotonic()-t)
        assert outputs==[0,0];rows.append(dict(start=start.tolist(),goal=goal.tolist(),rounded_goal=[9.,0.],original_result=outputs[0],certified_result=outputs[1],original_seconds=elapsed[0],certified_seconds=elapsed[1]))
    direct=[]
    for x in (8.96,8.97,8.98,8.99):
        start=np.asarray([x,0.]);goal=np.asarray([8.99,0.]);paths=[]
        for lib in libs:
            out=np.zeros((22,2));length=lib.astar(1.,0.,float('inf'),4.,20,start.ctypes.data_as(P),goal.ctypes.data_as(P),0,lens.ctypes.data_as(I),reserved.ctypes.data_as(P),len(walls),walls.ctypes.data_as(P),2.005,out.ctypes.data_as(P));paths.append(out[:length].copy())
        assert len(paths[0])==2 and np.array_equal(paths[0],paths[1]);direct.append(dict(start=start.tolist(),preserves_direct_initial_cell_case=True))
    write(HERE/'GRID_CERTIFICATE_UNIT_TEST.json',dict(passed=True,tests=rows,direct_initial_cell_tests=direct,goal_cell_infeasible_not_physical_goal_unreachable=True))
if __name__=='__main__':main()
