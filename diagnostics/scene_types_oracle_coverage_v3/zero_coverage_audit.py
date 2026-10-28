"""Read-only evidence audit and auxiliary geometric joint-path construction.

This never calls env.step, a controller, safety.project or the rollout runner.
Paths are diagnostic geometric witnesses, not C1/paper performance outcomes.
"""
from core import *
from scenes import geometry
from bottleneck_family.scenario import segment_distance
from single_integrator.environment import point_segment_distance
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import time

OUT=HERE/'zero_coverage_audit'


class SerialPaths:
    def __init__(self,s):
        self.s=s;self.geom=geometry(s);self.h=.035
        c=s['config'];self.wall=c['agent_radius']+c['wall_radius']+c['wall_collision_margin']
        self.pair=2*c['agent_radius']+c['agent_collision_margin']+1e-4
        self.cushion=self.h*np.sqrt(2)/2+1e-4
        x0,y0,x1,y1=s['bounds']
        xx=np.arange(np.ceil(x0/self.h),np.floor(x1/self.h)+1)*self.h
        yy=np.arange(np.ceil(y0/self.h),np.floor(y1/self.h)+1)*self.h
        grid=np.stack(np.meshgrid(xx,yy,indexing='ij'),-1)
        p=grid.reshape(-1,2)
        gap=point_segment_distance(p[:,None],self.geom.walls[None,:,0],self.geom.walls[None,:,1]).min(1)
        valid=self.geom.valid_points(p)&(gap>self.wall+self.cushion)
        ids=np.full(grid.shape[:2],-1,int);ids.reshape(-1)[valid]=np.arange(valid.sum())
        self.nodes=p[valid];self.tree=cKDTree(self.nodes);self.E=[];self.F=[];self.W=[]
        for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
            ia=np.arange(max(0,-dx),min(len(xx),len(xx)-dx))
            ja=np.arange(max(0,-dy),min(len(yy),len(yy)-dy))
            aa=ids[np.ix_(ia,ja)].ravel();bb=ids[np.ix_(ia+dx,ja+dy)].ravel();v=(aa>=0)&(bb>=0)
            self.E.extend(aa[v]);self.F.extend(bb[v]);self.W.extend([self.h*np.hypot(dx,dy)]*int(v.sum()))
        self.E=np.asarray(self.E);self.F=np.asarray(self.F);self.W=np.asarray(self.W)

    def visible(self,a,b,others):
        if not self.geom.visible(a,b):return False
        return not len(others) or bool(np.min(point_segment_distance(others,a,b))>self.pair+1e-6)

    def path(self,a,b,others):
        if self.visible(a,b,others):return np.array([a,b])
        free=np.ones(len(self.nodes),bool)
        if len(others):free=np.linalg.norm(self.nodes[:,None]-others[None],axis=-1).min(1)>self.pair+self.cushion
        edge=free[self.E]&free[self.F];ee=self.E[edge]+2;ff=self.F[edge]+2;ww=self.W[edge]
        src=list(ee)+list(ff);dst=list(ff)+list(ee);cost=list(ww)+list(ww)
        for index,q in enumerate((a,b)):
            ds,ns=self.tree.query(q,k=min(96,len(self.nodes)))
            linked=0
            for distance,j in zip(ds,ns):
                if not free[j]or distance>.35:continue
                if self.visible(q,self.nodes[j],others):
                    src.extend([index,int(j)+2]);dst.extend([int(j)+2,index]);cost.extend([distance,distance]);linked+=1
            if not linked:return None
        graph=coo_matrix((cost,(src,dst)),shape=(len(self.nodes)+2,)*2).tocsr()
        dist,pred=dijkstra(graph,indices=0,return_predecessors=True)
        if not np.isfinite(dist[1]):return None
        ids=[1];j=1
        while j!=0:j=int(pred[j]);ids.append(j)
        coords=np.vstack([a,b,self.nodes]);raw=coords[ids[::-1]];simple=[raw[0]];ix=0
        while ix<len(raw)-1:
            j=next(j for j in range(len(raw)-1,ix,-1)if self.visible(raw[ix],raw[j],others))
            simple.append(raw[j]);ix=j
        return np.asarray(simple)

    def solve(self,state,attempts=12):
        starts=np.asarray(state['positions']);goals=np.asarray(state['goals']);n=len(starts)
        rng=np.random.default_rng([20261028,self.s['type_id'],self.s['level'],state['index']])
        best=None
        for trial in range(attempts):
            p=starts.copy();remaining=list(range(n));moves=[]
            if trial==0:priority=np.argsort(-abs(goals[:,0]))
            elif trial==1:priority=np.argsort(abs(goals[:,0]))
            else:priority=rng.permutation(n)
            while remaining:
                found=False
                for i in priority:
                    if i not in remaining:continue
                    others=np.delete(p,i,axis=0);path=self.path(p[i],goals[i],others)
                    if path is None:continue
                    moves.append(dict(agent=int(i),path=path.tolist(),purpose='terminal_goal'))
                    p[i]=goals[i];remaining.remove(i);found=True;break
                if not found:break
            if remaining:continue
            result=self.verify(state,moves);result['priority_trial']=trial
            if best is None or result['serial_seconds_at_speed_limit']<best['serial_seconds_at_speed_limit']:best=result
            if best['serial_within_original_horizon']:break
        return best

    def verify(self,state,moves):
        p=np.asarray(state['positions']).copy();pair_min=float('inf');wall_min=float('inf');steps=0;distance=0.
        c=self.s['config']
        for move in moves:
            i=move['agent'];path=np.asarray(move['path']);assert np.array_equal(path[0],p[i])
            others=np.delete(p,i,axis=0)
            # Stationary pairs and every moving segment are both checked.
            ii,jj=np.triu_indices(len(p),1)
            pair_min=min(pair_min,float(np.linalg.norm(p[ii]-p[jj],axis=1).min())-self.pair)
            wall_min=min(wall_min,float(point_segment_distance(p[:,None],self.geom.walls[None,:,0],self.geom.walls[None,:,1]).min())-self.wall)
            for a,b in zip(path[:-1],path[1:]):
                pair_min=min(pair_min,float(point_segment_distance(others,a,b).min())-self.pair)
                wall_min=min(wall_min,float(segment_distance(a,b,self.geom.walls[:,0],self.geom.walls[:,1]).min())-self.wall)
                assert self.geom.visible(a,b)
                length=float(np.linalg.norm(b-a));distance+=length;steps+=int(np.ceil(length/(c['max_speed']*c['dt'])))
            p[i]=path[-1]
        assert np.array_equal(p,np.asarray(state['goals']))
        assert pair_min>0 and wall_min>0
        return dict(geometric_joint_witness=True,verification='all continuous straight-segment sweeps against actual walls and all stationary robots; exact terminal goals',
            minimum_pair_clearance=pair_min,minimum_wall_clearance=wall_min,moves=moves,total_distance=distance,
            serial_seconds_at_speed_limit=steps*c['dt'],serial_within_original_horizon=steps<=c['max_steps'],
            horizon_seconds=c['max_steps']*c['dt'],original_CBF_verified=False,original_solver_replayed=False,
            scope='Geometric construction with speed-limit retiming only. Not an executed policy outcome or proof of C1-CBF-admissible velocity sequence.')


def main():
    OUT.mkdir(exist_ok=True);cover=read(HERE/'COVERAGE.json')
    zero={(x['type_id'],x['level'])for x in cover['scenes']if x['families'][4]['coverage']['64']==0}
    old=read(ROOT/'diagnostics/scene_types_basis_dimension_v1/CALIBRATION.json')
    hist=[]
    for t,level in sorted(zero):
        for h in old['history'].get(str(t),[]):
            if h['scene']['level']==level and h['scene']['widen']==0:
                hist.append(dict(type_id=t,level=level,source=str(ROOT/'diagnostics/scene_types_basis_dimension_v1/CALIBRATION.json'),
                    compatible_geometry_and_initial_states=True,horizon_differs=True,astar_success=h['metrics']['witness_success'],
                    astar_n=h['metrics']['witness_n'],witness_status=h['metrics']['witness_status']))
    dump(OUT/'HISTORY_CHECK.json',dict(sources=hist,evidence_gap='No successful old joint A* witness on these exact zero-coverage ladder scenes. Single-agent paths/gates do not certify joint solvability.',
        new_work='Auxiliary serial grid-path search and continuous segment verification only; zero environment steps, no expensive rollout, no controller or safety changes.',
        zero_scenes_snapshot=sorted(zero),frozen_scenes_sha256=sha(HERE/'SCENES.json')))
    results=[]
    for e in read(HERE/'SCENES.json')['entries']:
        if(e['type_id'],e['level'])not in zero:continue
        s=e['scene'];planner=SerialPaths(s)
        for state in s['states'][:8]:
            path=OUT/f"t{e['type_id']:02d}_l{e['level']}_s{state['index']}.json"
            if path.exists():result=read(path)
            else:
                start=time.monotonic();v=planner.solve(state)
                result=dict(type_id=e['type_id'],level=e['level'],state_index=state['index'],scene=s['name'],
                    scene_sha256=s['definition_sha256'],search_seconds=time.monotonic()-start,
                    found=v is not None,witness=v,not_found_interpretation='Finite diagnostic serial-grid search did not find a path; not infeasibility.')
                dump(path,result)
            results.append(result)
            print('GEOMETRIC_WITNESS',e['type_id'],e['level'],state['index'],result['found'],
                  result['witness']['serial_seconds_at_speed_limit']if result['found']else None,flush=True)
            dump(OUT/'RESULTS.json',dict(cases=results,environment_steps=0,rollout_count=0,complete=False))
    dump(OUT/'RESULTS.json',dict(cases=results,environment_steps=0,rollout_count=0,complete=True,
        interpretation='Independent auxiliary geometric witnesses. These do not count toward any Oracle or baseline success, and have not been replayed through the original safety solver.'))


if __name__=='__main__':main()
