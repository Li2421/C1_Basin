"""Direct native differential checks,including real bounded-cache fallback."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import ctypes,numpy as np,subprocess

PTR=ctypes.POINTER(ctypes.c_double);IPTR=ctypes.POINTER(ctypes.c_int)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});os.nice(10)
    verify_receipt('PA_STATIONARY_EDGE_CANDIDATE_20261022.receipt.json')
    cpp=HERE/'astar_kernel_stationary_edge_candidate_20261022.cpp';text=cpp.read_text();assert text.count('stationary_edges.size()<65536')==1
    capped=HERE/'astar_kernel_stationary_edge_cap4_test_20261022.cpp';assert not capped.exists();capped.write_text(text.replace('stationary_edges.size()<65536','stationary_edges.size()<4'))
    so=capped.with_suffix('.so');subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off',str(capped),'-o',str(so)],check=True)
    libs=[ctypes.CDLL(str(HERE/name)) for name in ('astar_kernel_tiled.so','astar_kernel_stationary_edge_candidate_20261022.so',so.name)]
    signature=[ctypes.c_double]*4+[ctypes.c_int,PTR,PTR,ctypes.c_int,IPTR,PTR,ctypes.c_int,PTR,ctypes.c_double,PTR]
    for lib in libs:lib.astar.argtypes=signature;lib.astar.restype=ctypes.c_int
    checks=0
    def check(walls,margin,sep,res,horizon,start,goal,reserved):
        nonlocal checks
        start=np.ascontiguousarray(start,np.float64);goal=np.ascontiguousarray(goal,np.float64);walls=np.ascontiguousarray(walls,np.float64);lens=np.asarray([len(p) for p in reserved],np.int32);refs=np.ascontiguousarray(np.concatenate(reserved) if reserved else np.zeros((0,2)),np.float64)
        paths=[]
        for lib in libs:
            out=np.zeros((horizon+2,2),np.float64)
            n=lib.astar(res,0.,float('inf'),sep,horizon,start.ctypes.data_as(PTR),goal.ctypes.data_as(PTR),len(lens),lens.ctypes.data_as(IPTR),refs.ctypes.data_as(PTR),len(walls),walls.ctypes.data_as(PTR),margin,out.ctypes.data_as(PTR))
            paths.append((n,out[:n].tobytes()))
        assert paths[0]==paths[1]==paths[2],(checks,start,goal,reserved)
        checks+=1
    box=np.asarray([[[-1,-1],[1,-1]],[[1,-1],[1,1]],[[1,1],[-1,1]],[[-1,1],[-1,-1]]],float);rng=np.random.default_rng(2026102211)
    for index in range(128):
        res=(.1,.2,.25)[index%3];horizon=(8,12,20,28)[index%4];walls=box if index%5 else np.concatenate([box,[[[.04,-.8],[.04,.8]]]]);margin=.04+(index%3)*.005
        start=rng.uniform(-.65,.65,2);goal=rng.uniform(-.65,.65,2)
        if index%7==0:goal=start+res*.05
        reserved=[rng.uniform(-.7,.7,((index+j)%4+1,2)) for j in range(index%4)]
        for delta,ref_delta in ((0.,0.),(1e-14,0.),(1e-9,1e-14),(.07,.15),(0.,0.)):
            refs=[p.copy() for p in reserved]
            for p in refs:p[0]+=ref_delta
            check(walls,margin,.08+(index%3)*.003,res,horizon,start+delta,goal,refs)
    for start in ([-0.,+0.],[+0.,-0.],[-0.,-0.],[+0.,+0.]):
        for refs in ([],[np.asarray([[.3,.3]])],[np.asarray([[.11,.11],[.3,.3]])]):check(box,.04,.08,.2,12,start,[.4,.4],refs)
    for delta in (0.,1e-14,1e-9,0.):check(box,.04,.08,.2,12,[.011+delta,.017-delta],[.4,.4],[np.asarray([[.4,.4]])])
    stats=[]
    for lib in libs[1:]:
        lib.pa_stationary_edge_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)];a=(ctypes.c_uint64*8)();lib.pa_stationary_edge_stats(a);stats.append(list(a))
    assert stats[0][2]>0 and stats[0][7]<=65536
    assert stats[1][5]>0 and stats[1][7]<=4,stats
    write(HERE/'PA_STATIONARY_EDGE_FINITE_20261022.json',dict(utc=utc(),passed=True,bitwise_path_checks=checks,native_versions_compared=3,query_memo_bypassed=True,candidate_stats=stats[0],cap4_stats=stats[1],capacity_fallback_actually_exercised=True,cap4_only_source_change='65536->4 in pure compute cache capacity',new_research_rollouts=0,environment_steps=0))
    print('PASS',checks,'direct native path checks;bounded fallback',stats,flush=True)


if __name__=='__main__':main()
