"""Static feature tests, no environment stepping or outcome labels."""
from common import *
from selector_features40 import *
from dataclasses import asdict

def main():
    from bottleneck_family.scenario import Config
    cfg=asdict(Config(num_agents=4,half_length=2.,half_height=1.));m=StaticMap(cfg,rectangles=[])
    p=np.array([[-1.5,-.3],[1.5,.3],[-1.5,.3],[1.5,-.3]]);g=-p;v=np.zeros_like(p)
    assert m.valid(p).all();grid=OccupancyGrid(m);path,length=grid.path(p[0],g[0]);assert np.array_equal(path[0],p[0]) and np.array_equal(path[-1],g[0]);assert np.linalg.norm(g[0]-p[0])<=length<=np.linalg.norm(g[0]-p[0])+SPACING*10
    phys=physical12(p,v,g,m);features=grid.features(p,g,400,.05);assert phys.shape==(12,) and features.shape==(7,) and np.isfinite(features).all()
    assert features[0]>=1 and 0<features[1]<1 and 0<=features[4]<=6
    for arm in ['F10','F13','F15','S13','S15','Sprogress','right_hand']:
        a=action9(arm);assert a.shape==(9,) and np.isfinite(a).all();assert len(np.r_[phys,features,np.zeros(5),a])==len(FEATURE_NAMES)==33
    assert action9('S13')[6]==np.float32(5/7);assert action9('Sprogress')[7]==1;assert action9('right_hand')[8]==1
    X=np.array([[1,np.nan,3],[3,np.nan,3]],np.float32);tr=fit_transform(X);assert np.array_equal(tr['mean'],[2,0,3]);assert np.isfinite(apply_transform(X,tr)).all()
    from ring_exchange.environment import Config as RingConfig
    ring=StaticMap(asdict(RingConfig()),scene='ring_exchange');assert not ring.valid(np.zeros((1,2)))[0];assert ring.valid(np.array([[1.8,0.]])).all();assert not ring.links(np.array([[1.8,0.]]),np.array([[-1.8,0.]]))[0]
    from double_bottleneck.environment import Config as DoubleConfig
    double=StaticMap(asdict(DoubleConfig()),scene='double_bottleneck');assert double.valid(np.array([[0.,0.]])).all();assert not double.valid(np.array([[1.3,.6]]))[0]
    from four_way_intersection.environment import Config as FourConfig
    four=StaticMap(asdict(FourConfig()),scene='four_way_intersection');assert four.valid(np.zeros((1,2))).all()
    write(HERE/'SELECTOR_FEATURES40_UNIT_TEST.json',dict(utc=utc(),passed=True,dimensions=33,feature_names=FEATURE_NAMES,source_sha256=sha(HERE/'selector_features40.py'),tests=16,physical_steps=0,task_outcomes_read=0))

if __name__=='__main__':main()
