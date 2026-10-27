"""Rigid-coordinate extension of author rectangles to rotated benchmark walls."""
import inspect,types
import numpy as np

def install():
    from PyQt5.QtCore import QPointF,QRectF
    import collision
    old_closest=collision.get_closest_point_on_rect
    def closest(point,rect):
        if not hasattr(rect,'basis'):return old_closest(point,rect)
        p=rect.basis.T@(np.array([point.x(),point.y()])-rect.origin)
        q=old_closest(QPointF(*p),rect.local)
        return QPointF(*(rect.origin+rect.basis@np.array([q.x(),q.y()])))
    collision.get_closest_point_on_rect=closest
    source=inspect.getsource(collision.check_line_collision_with_obstacles)
    anchor='            rect_edges = ['
    assert source.count(anchor)==1
    source=source.replace(anchor,'            if hasattr(rect, "corners"):\n                rect_points = rect.corners\n'+anchor)
    exec(compile(source,collision.__file__+':rigid_rectangle_extension','exec'),collision.__dict__)
    class RotatedRect:
        def __init__(self,a,b,width):
            self.origin=np.asarray(a);d=np.asarray(b)-a;length=np.linalg.norm(d);u=d/length
            self.basis=np.stack([u,[-u[1],u[0]]],axis=1);self.local=QRectF(0.,-width,length,2*width)
            self.corners=[self.origin+self.basis@p for p in ([0,-width],[length,-width],[length,width],[0,width])]
        def center(self):return QPointF(*(self.origin+self.basis@np.array([self.local.width()/2,0.])))
        def width(self):return self.local.width()
        def height(self):return self.local.height()
        def left(self):return self.local.left()
        def right(self):return self.local.right()
        def top(self):return self.local.top()
        def bottom(self):return self.local.bottom()
        def contains(self,p):
            q=self.basis.T@(np.array([p.x(),p.y()])-self.origin);return self.local.contains(QPointF(*q))
    return RotatedRect

def placement_map_extension(sim):
    """Identical rectangle inflation and pixel rounding, in its local axes."""
    original=sim.updatePlacementMap
    def update(self,radius,index,increment,map_safety_margin):
        rotated=[o for o in self.static_obstacles if hasattr(getattr(o,'rect',None),'basis')]
        if not rotated:return original(radius,index,increment,map_safety_margin)
        saved=self.static_obstacles;self.static_obstacles=[o for o in saved if o not in rotated]
        try:original(radius,index,increment,map_safety_margin)
        finally:self.static_obstacles=saved
        margin=(int(radius*self.scale)+int(increment*self.scale)+int(map_safety_margin*self.scale))/self.scale
        yy,xx=np.ogrid[:self.sim_height,:self.sim_width]
        for ob in rotated:
            r=ob.rect;dx=xx/self.scale-r.origin[0];dy=yy/self.scale-r.origin[1]
            x=r.basis[0,0]*dx+r.basis[1,0]*dy;y=r.basis[0,1]*dx+r.basis[1,1]*dy
            mask=(x>=r.left()-margin)&(x<=r.right()+margin)&(y>=r.top()-margin)&(y<=r.bottom()+margin)
            self.placement_map[index][mask]=0
    sim.updatePlacementMap=types.MethodType(update,sim)
