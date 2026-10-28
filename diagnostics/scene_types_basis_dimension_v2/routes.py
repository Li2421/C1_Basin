"""Goal-input adapter; the shortest-path algorithm is the original Geometry.path."""
from core import *
from bottleneck_family.scenario import Geometry

class SegmentRouteGeometry:
    """Supply the existing free/legacy-segment geometry to the original algorithm.

    No walls are added. Endpoint offset nodes are only needed for the legacy
    angled segment map, whose native geometry has no path method. The new
    rectangular maps use their original Geometry and original corner roadmap.
    """
    path=Geometry.path
    _graph=Geometry._graph
    def __init__(self,geom):
        self.geom=geom;self.walls=geom.walls;self.margin=geom.margin
        if not len(self.walls):self.nodes=np.zeros((0,2));self.graph=[];return
        ends=np.unique(self.walls.reshape(-1,2),axis=0)
        angles=np.arange(16)*2*np.pi/16
        offsets=1.5*self.margin*np.stack([np.cos(angles),np.sin(angles)],axis=1)
        nodes=(ends[:,None]+offsets).reshape(-1,2)
        self.nodes=nodes[self.valid_points(nodes)];self.graph=self._graph(self.nodes)
    def valid_points(self,p):return self.geom.valid_points(p)
    def visible(self,a,b):return self.geom.visible(a,b)

def route_geometry(geom):return geom if hasattr(geom,'path')else SegmentRouteGeometry(geom)

def guidance(geom,positions,true_goals,enabled):
    if not enabled:
        return np.asarray(true_goals).copy(),np.linalg.norm(true_goals-positions,axis=-1),None
    guides=[];remaining=[];paths=[]
    for p,g in zip(positions,true_goals):
        path=geom.path(p,g)
        if path is None:raise ValueError('No current single-robot route under the original geometry')
        length=np.linalg.norm(np.diff(path,axis=0),axis=1);remaining.append(float(length.sum()));paths.append(path)
        if geom.visible(p,g):guides.append(g.copy());continue
        arc=np.r_[0.,np.cumsum(length)];candidates=[]
        for j in range(1,len(path)):
            if arc[j]<=1.:candidates.append((arc[j],path[j]))
            elif arc[j-1]<1.:
                q=path[j-1]+(1.-arc[j-1])/length[j-1]*(path[j]-path[j-1]);candidates.append((1.,q))
                break
        visible=[(s,q)for s,q in candidates if geom.visible(p,q)]
        if not visible:raise ValueError('Route exists but no positive visible guide within the 1m lookahead')
        guides.append(max(visible,key=lambda item:item[0])[1])
    return np.asarray(guides),np.asarray(remaining),paths
