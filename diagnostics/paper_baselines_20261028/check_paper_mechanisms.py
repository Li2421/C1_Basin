"""Meaningful numerical and geometry regression before paper rollouts."""
import sys,json,time
from pathlib import Path
import numpy as np
from formula_baselines import risk_aux,Wang
HERE=Path(__file__).resolve().parent

def main():
    rng=np.random.default_rng(20261028);err=0.
    for _ in range(100):
        x,g,c=rng.normal(size=(3,2));theta=rng.normal();r=.1;val,grad,gt,*_=risk_aux(x,g,theta,c,r)
        h=1e-6;fd=np.array([(risk_aux(x+np.eye(2)[k]*h,g,theta,c,r)[0]-risk_aux(x-np.eye(2)[k]*h,g,theta,c,r)[0])/(2*h)for k in range(2)])
        ft=(risk_aux(x,g,theta+h,c,r)[0]-risk_aux(x,g,theta-h,c,r)[0])/(2*h)
        err=max(err,np.max(np.abs(grad-fd))/(1+np.max(np.abs(grad))),abs(gt-ft)/(1+abs(gt)))
    assert err<1e-7,err
    w=Wang(None,0);u=np.zeros(2);v=np.zeros(2)
    # Interior feasible wedge and nonzero nominal: vertex/edge/empty-interior cases.
    assert w.classify(np.array([[-1.,0.],[0.,-1.]]),np.zeros(2),u,np.array([-1.,-1.]),v)==1
    assert w.classify(np.array([[-1.,0.]]),np.zeros(1),u,np.array([-1.,0.]),v)==2
    assert w.classify(np.array([[-1.,0.],[1.,0.]]),np.zeros(2),u,np.array([-1.,0.]),v)==3
    from paper_adapters import vendor_path
    vendor_path('mgr')
    from mgr_geometry import install
    Rotated=install()
    import collision
    from PyQt5.QtCore import QPointF,QRectF
    from obstacles import RectObstacle
    from types import SimpleNamespace
    axis=RectObstacle(0.,-.05,1.,.1);angle=.65;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    origin=np.array([2.,3.]);rr=Rotated(origin,origin+Q@np.array([1.,0.]),.05);rot=SimpleNamespace(type='rect',rect=rr)
    geoerr=0.
    for _ in range(50):
        p=rng.normal(size=2);a=collision.get_closest_point_on_obstacle(QPointF(*p),axis);b=collision.get_closest_point_on_obstacle(QPointF(*(origin+Q@p)),rot)
        geoerr=max(geoerr,np.linalg.norm(np.array([b.x(),b.y()])-(origin+Q@np.array([a.x(),a.y()]))))
    assert geoerr<1e-12,geoerr
    for a,b in [(np.array([.5,-1.]),np.array([.5,1.])),(np.array([-1.,1.]),np.array([2.,1.]))]:
        ca=collision.check_line_collision_with_obstacles(QPointF(*a),QPointF(*b),.5,[axis],5.,.1)[0]
        cb=collision.check_line_collision_with_obstacles(QPointF(*(origin+Q@a)),QPointF(*(origin+Q@b)),.5,[rot],5.,.1)[0]
        assert ca==cb
    result=dict(risk_gradient_relative_error=err,wang_types=[1,2,3],mgr_rigid_geometry_error=geoerr,passed=True,limitations='Checks validate specific derivatives/branch geometry; not full paper fidelity or closed-loop guarantees.')
    (HERE/'mechanism_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
