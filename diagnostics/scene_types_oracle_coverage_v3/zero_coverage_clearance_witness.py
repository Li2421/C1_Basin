"""Try a clearance-preserving geometric recipe after shortest paths were slow.

Only diagnostic witnesses. No Oracle candidates, scene or controller changes.
"""
from zero_coverage_audit import SerialPaths,point_segment_distance,segment_distance
from core import *
from scenes import geometry
import zero_coverage_cbf_check as cbf
import zero_coverage_schedule as scheduling

OUT=HERE/'zero_coverage_audit/clearance_variant'


class ClearancePaths(SerialPaths):
    def visible(self,a,b,others):
        if not super().visible(a,b,others):return False
        endpoint=float(point_segment_distance(np.array([a,b])[:,None],self.geom.walls[None,:,0],self.geom.walls[None,:,1]).min())
        wall_target=min(self.wall+.020,endpoint-1e-6)
        gap=float(segment_distance(a,b,self.geom.walls[:,0],self.geom.walls[:,1]).min())
        if gap<wall_target:return False
        if len(others):
            endpoint_pair=float(np.linalg.norm(np.array([a,b])[:,None]-others[None],axis=-1).min())
            if point_segment_distance(others,a,b).min()<min(self.pair+.020,endpoint_pair-1e-6):return False
        return True


def main():
    OUT.mkdir(exist_ok=True);scheduling.OUT=OUT
    history=dict(previous_source=str(HERE/'zero_coverage_audit/CBF_RESULTS.json'),
        gap='Previous geometric paths have slow CBF retiming and no time-limit witness. Try explicit 0.02m soft clearance for shortcutting, with fixed-endpoint exception.',
        recipe='Sequential grid paths as before; accepted segments retain min(0.02m added clearance, endpoint clearance minus1e-6). Same exact geometry/CBF/timestep/speed. Existing found states first, stop within each ladder once a certified in-horizon witness exists.',
        environment_steps=0,solver_calls=0)
    dump(OUT/'HISTORY_CHECK.json',history)
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']};results=[]
    for level in range(3):
        s=scenes[(8,level)];planner=ClearancePaths(s)
        found_states=[i for i in range(8)if read(HERE/'zero_coverage_audit'/f't08_l{level}_s{i}.json')['found']]
        others=[i for i in range(8)if i not in found_states]
        for i in found_states+others:
            source=OUT/f't08_l{level}_s{i}.json';cpath=source.with_name(source.stem+'_cbf.json');target=cpath.with_name(cpath.stem+'_scheduled.json')
            if target.exists():r=read(target)
            else:
                witness=planner.solve(s['states'][i],attempts=4)
                dump(source,dict(type_id=8,level=level,state_index=i,found=witness is not None,witness=witness))
                if witness is None:print('CLEARANCE_NO_GEOMETRIC_PATH',level,i,flush=True);continue
                try:
                    r,p,u=cbf.retime(s,s['states'][i],witness)
                    trace=cpath.with_suffix('.npz');np.savez_compressed(trace,positions=p,actions=u)
                    r.update(type_id=8,level=level,state_index=i,trace=str(trace));dump(cpath,r)
                    r=scheduling.schedule(cpath,s);dump(target,r)
                except Exception as exc:
                    dump(OUT/f't08_l{level}_s{i}_error.json',dict(error=str(exc)));print('CLEARANCE_UNKNOWN',level,i,str(exc),flush=True);continue
            results.append(r);print('CLEARANCE_JOINT_CBF',level,i,r['seconds'],r['within_original_horizon'],flush=True)
            dump(OUT/'RESULTS.json',dict(cases=results,complete=False,environment_steps=0,solver_calls=0))
            if r['within_original_horizon']:break
    dump(OUT/'RESULTS.json',dict(cases=results,complete=True,environment_steps=0,solver_calls=0,
        meaning='Auxiliary finite search for constructive joint witnesses, not an expert success-rate estimate or Oracle result.'))


if __name__=='__main__':main()
