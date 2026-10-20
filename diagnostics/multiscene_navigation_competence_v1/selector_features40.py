"""Outcome-free physical, occupancy-grid geometry, behavior and action features."""
from common import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from single_integrator.environment import point_segment_distance,segment_distance
from shared_rollout_db.src.rollout_db import uid

SPACING=.05
FEATURE_NAMES=['N','goal_distance_mean','goal_distance_std','goal_distance_min','goal_distance_max','pair_min','pair_mean','wall_min','wall_mean','speed_mean','speed_max','L_over_radius','narrowest_effective_width_over_diameter','free_fraction','bottleneck_count_mean','bottleneck_count_max','crossing_or_shared_bottleneck_pairs','path_length_mean_over_budget','path_length_max_over_budget','probe_positive_progress','probe_stall40','probe_min_pair_over_radius','probe_mean_speed','probe_projection_fraction','early_eta1','early_eta2','early_eta3','late_eta1','late_eta2','late_eta3','switch_fraction','progress_switch','continuing_right_hand']

class StaticMap:
    def __init__(self,config,rectangles=None,scene=None):
        self.config=config;self.radius=float(config['agent_radius']);self.speed=float(config['max_speed']);self.scene=scene
        self.wall_radius=float(config.get('wall_radius',0));self.extra=float(config.get('wall_collision_margin',config.get('collision_margin',0)))
        self.rectangles=None;self.native_outside=None
        if 'outer_radius' in config:
            self.bounds=(-config['outer_radius'],-config['outer_radius'],config['outer_radius'],config['outer_radius']);self.walls=None
        elif rectangles is not None:
            from geometry import GeneratedGeometry
            from bottleneck_family.scenario import Config
            g=GeneratedGeometry(Config(**config),rectangles);self.walls=g.walls;self.rectangles=g.rectangles
            self.bounds=(-config['half_length'],-config['half_height'],config['half_length'],config['half_height'])
        else:
            if scene=='double_bottleneck':
                from double_bottleneck.environment import Config,DoubleBottleneckEnv
                env=DoubleBottleneckEnv(Config(**config));length=config['world_half_length'];height=max(config['outer_half_height'],config['chamber_half_height']);self.bounds=(-length,-height,length,height);self.native_outside=env.outside
            elif scene=='four_way_intersection':
                from four_way_intersection.environment import Config,FourWayIntersectionEnv
                env=FourWayIntersectionEnv(Config(**config));h=config['world_half_extent'];self.bounds=(-h,-h,h,h)
            else:raise ValueError('Unrecognized static geometry')
            self.walls=env.walls
    def clearance(self,p):
        p=np.asarray(p,float)
        if self.walls is None:
            rr=np.linalg.norm(p,axis=-1);return np.minimum(rr-self.config['obstacle_radius'],self.config['outer_radius']-rr)-self.radius
        return point_segment_distance(p[...,None,:],self.walls[:,0],self.walls[:,1]).min(-1)-self.radius-self.wall_radius
    def valid(self,p):
        p=np.asarray(p,float);x0,y0,x1,y1=self.bounds
        result=(self.clearance(p)>self.extra)&(p[...,0]>x0)&(p[...,0]<x1)&(p[...,1]>y0)&(p[...,1]<y1)
        if self.rectangles is not None:
            q=self.rectangles;inside=((p[...,0,None]>=q[:,0])&(p[...,0,None]<=q[:,2])&(p[...,1,None]>=q[:,1])&(p[...,1,None]<=q[:,3])).any(-1);result&=~inside
        if self.native_outside is not None:result&=~self.native_outside(p.reshape(-1,2)).reshape(result.shape)
        return result
    def links(self,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        if self.walls is None:
            # Exact minimum swept radius and convex outer-circle endpoint test.
            minimum=point_segment_distance(np.zeros(2),a,b)
            return (minimum>self.config['obstacle_radius']+self.radius+self.extra)&(np.maximum(np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1))<self.config['outer_radius']-self.radius-self.extra)
        return (segment_distance(a[...,None,:],b[...,None,:],self.walls[:,0],self.walls[:,1]).min(-1)>self.radius+self.wall_radius+self.extra)&self.valid(a)&self.valid(b)

def physical12(p,v,g,geometry):
    p=np.asarray(p,float);v=np.asarray(v,float);g=np.asarray(g,float);r=geometry.radius
    L=max(2*np.linalg.norm(g-g.mean(0),axis=-1).max(),2*r);d=np.linalg.norm(g-p,axis=-1)/L;ii,jj=np.triu_indices(len(p),1);pair=np.linalg.norm(p[ii]-p[jj],axis=-1)/(2*r);wall=geometry.clearance(p)/r;vel=np.linalg.norm(v,axis=-1)/geometry.speed
    return np.asarray([len(p),d.mean(),d.std(),d.min(),d.max(),pair.min(),pair.mean(),wall.min(),wall.mean(),vel.mean(),vel.max(),L/r],np.float32)

class OccupancyGrid:
    def __init__(self,geometry):
        self.geometry=geometry;x0,y0,x1,y1=geometry.bounds
        self.x=np.linspace(x0,x1,round((x1-x0)/SPACING)+1);self.y=np.linspace(y0,y1,round((y1-y0)/SPACING)+1)
        xx,yy=np.meshgrid(self.x,self.y);self.points=np.stack([xx,yy],-1);self.shape=xx.shape;p=self.points.reshape(-1,2);self.free=geometry.valid(p).reshape(xx.shape);ids=np.arange(len(p)).reshape(xx.shape);rr=[];cc=[];vv=[]
        for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
            ya=slice(0,self.shape[0]-dy);yb=slice(dy,self.shape[0]);xa=slice(max(0,-dx),min(self.shape[1],self.shape[1]-dx));xb=slice(max(0,dx),min(self.shape[1],self.shape[1]+dx));a=ids[ya,xa].ravel();b=ids[yb,xb].ravel();ok=self.free.ravel()[a]&self.free.ravel()[b];a=a[ok];b=b[ok];ok=geometry.links(p[a],p[b]);a=a[ok];b=b[ok];w=np.full(len(a),SPACING*np.hypot(dx,dy));rr.extend([a,b]);cc.extend([b,a]);vv.extend([w,w])
        self.graph=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(p),len(p))).tocsr()
    def path(self,start,goal):
        p=self.points.reshape(-1,2);n=len(p);connect=[]
        for q in [start,goal]:
            dist=np.linalg.norm(p-q,axis=-1);ix=np.flatnonzero(self.free.ravel()&(dist<=.1));ix=ix[self.geometry.links(np.broadcast_to(q,(len(ix),2)),p[ix])]
            if not len(ix):raise ValueError('No valid occupancy-grid endpoint connection')
            connect.append((ix,dist[ix]))
        base=self.graph.tocoo();rr=[base.row];cc=[base.col];vv=[base.data]
        for k,(ix,w) in enumerate(connect):rr.extend([ix,np.full(len(ix),n+k)]);cc.extend([np.full(len(ix),n+k),ix]);vv.extend([w,w])
        graph=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n+2,n+2)).tocsr();distance,predecessor=dijkstra(graph,directed=False,indices=n,return_predecessors=True)
        if not np.isfinite(distance[n+1]):raise ValueError('No occupancy-grid path; feature gap')
        chain=[n+1]
        while chain[-1]!=n:chain.append(int(predecessor[chain[-1]]))
        points=np.vstack([p,start,goal]);return points[chain[::-1]],float(distance[n+1])
    def effective_widths(self,path):
        # Quantized transverse center-space spans plus one robot diameter.
        q=path[1:-1];direction=path[2:]-path[:-2];normal=np.column_stack([-direction[:,1],direction[:,0]]);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-30)
        if not len(q):q=path[:1];normal=np.array([[0.,1.]])
        spans=[];maximum=int(np.ceil(np.hypot(self.x[-1]-self.x[0],self.y[-1]-self.y[0])/SPACING))
        for sign in [-1,1]:
            count=np.zeros(len(q),int);active=np.ones(len(q),bool)
            for k in range(1,maximum+1):
                probe=q+sign*k*SPACING*normal
                ij=np.rint((probe-np.array([self.x[0],self.y[0]]))/SPACING).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<len(self.x))&(ij[:,1]>=0)&(ij[:,1]<len(self.y));valid=np.zeros(len(q),bool);valid[inside]=self.free[ij[inside,1],ij[inside,0]];active&=valid;count[active]=k
                if not active.any():break
            spans.append(count*SPACING)
        return spans[0]+spans[1]+2*self.geometry.radius,q
    def features(self,starts,goals,H,dt):
        paths=[];lengths=[];widths=[];bottlenecks=[];counts=[];diameter=2*self.geometry.radius
        for p,g in zip(starts,goals):
            path,length=self.path(p,g);width,q=self.effective_widths(path);narrow=width<4*diameter
            paths.append(path);lengths.append(length);widths.append(width);bottlenecks.append(q[narrow]);counts.append(int(np.sum(narrow&~np.r_[False,narrow[:-1]])))
        pairs=0
        for i in range(len(paths)):
            for j in range(i):
                a,b=paths[i],paths[j];cross=segment_distance(a[:-1,None],a[1:,None],b[None,:-1],b[None,1:]).min()<=1e-10
                shared=len(bottlenecks[i]) and len(bottlenecks[j]) and (np.linalg.norm(bottlenecks[i][:,None]-bottlenecks[j][None,:],axis=-1).min()<=diameter)
                pairs+=bool(cross or shared)
        budget=self.geometry.speed*H*dt
        return np.asarray([min(w.min() for w in widths)/diameter,self.free.mean(),np.mean(counts),max(counts),pairs,np.mean(lengths)/budget,max(lengths)/budget],np.float32)

def action9(arm):
    exact=read(HERE/'ETA_EXACT.json');zero=[0.,0.,0.]
    if arm=='right_hand':return np.asarray(zero+zero+[1.,0.,1.],np.float32)
    if arm.startswith('F'):a=b=exact[arm[1:]]['values'];fraction=1.;progress=0.
    else:a=exact['15' if arm=='S15' else '13']['values'];b=exact['10']['values'];fraction=6/7 if arm=='Sprogress' else 5/7;progress=float(arm=='Sprogress')
    return np.asarray(a+b+[fraction,progress,0.],np.float32)

def fit_transform(X):
    with np.errstate(invalid='ignore'):mean=np.nanmean(X.astype(np.float64),axis=0)
    mean=np.where(np.isfinite(mean),mean,0);filled=np.where(np.isfinite(X),X,mean);scale=filled.std(0);scale=np.where(scale<1e-6,1,scale)
    return dict(mean=mean.astype(np.float32),scale=scale.astype(np.float32))
def apply_transform(X,tr):return ((np.where(np.isfinite(X),X,tr['mean'])-tr['mean'])/tr['scale']).astype(np.float32)
