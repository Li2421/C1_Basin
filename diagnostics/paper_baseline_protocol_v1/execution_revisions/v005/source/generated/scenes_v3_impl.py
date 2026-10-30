"""Geometry-first scene ladders. Never imports or observes correction outcomes."""
from core import *
from routes import route_geometry
from bottleneck_family.scenario import Config,Geometry
from scene_environment import bottleneck_geometry
from swap_geometry_user_fix_20261023 import GeometryRevision
from single_integrator.environment import point_segment_distance

SEED=2026102801
TITLES={1:'开阔·圆周对穿',2:'开阔·两股垂直交叉流',3:'宽走廊·同向换道',4:'封闭方形场地·随机起终点（拥挤）',5:'方岛环岛·四向',6:'带障碍的圆圈对穿',7:'散布障碍森林·随机起终点',8:'窄门·两组对向互换',9:'单向挤门（疏散）',10:'单车道长走廊·两组对向',11:'单车道走廊 + 中间让行湾',12:'L 形盲角·对向',13:'T 形路口·三向',14:'窄十字路口·四向对穿',15:'四房间连通（串联瓶颈）',16:'货架仓库·窄巷',17:'集结到小区域'}
LADDERS={1:[8,10,12],2:[4,5,6],3:[6,8,10],4:[10,11,12],5:[2,3],6:[.05,.055,.06],7:[6,8,10],8:[3,4,5],9:[6,8,10],10:[3,4,2],11:[2,3,1],12:[3,4,2],13:[2,3],14:[.40,.36,.46],15:[10,12,8],16:[10,12,8],17:[12,16,20]}

def rect_complement(bounds,free):
    """Exact rectangular decomposition outside a union of axis-aligned rooms."""
    x0,y0,x1,y1=bounds
    xs=sorted({x0,x1,*[np.clip(r[k],x0,x1)for r in free for k in (0,2)]})
    ys=sorted({y0,y1,*[np.clip(r[k],y0,y1)for r in free for k in (1,3)]})
    cells=[]
    for a,b in zip(xs[:-1],xs[1:]):
        run=None
        for c,d in zip(ys[:-1],ys[1:]):
            mid=((a+b)/2,(c+d)/2);occupied=not any(u<=mid[0]<=w and v<=mid[1]<=z for u,v,w,z in free)
            if occupied:
                if run is None:run=c
            elif run is not None:cells.append((a,run,b,c));run=None
        if run is not None:cells.append((a,run,b,ys[-1]))
    return cells

def bcgeom(n,r,hx,hy,rects=(),barrier=None):
    # Tiny near-boundary dummy slabs implement the documented Config workaround.
    if barrier is None:bx=(hx-max(.31,2*r+.04),);op=(((0.,2*hy-.004),),);th=.01
    else:bx,op,th=barrier
    c=Config(num_agents=n,half_length=hx,half_height=hy,barrier_x=bx,openings=op,
        barrier_thickness=th,extra_rectangles=tuple(rects),agent_radius=r,wall_radius=4/600,
        wall_collision_margin=.005,dt=.05,max_speed=.5,max_steps=1000,seed=SEED)
    return c.to_dict(),Geometry(c)

def separate(p,r):
    return len(p)<2 or bool(np.all(np.linalg.norm(np.asarray(p)[:,None]-np.asarray(p)[None],axis=-1)[np.triu_indices(len(p),1)]>=2*r+.05))

def sample(rng,geom,n,r,region,avoid=None,mintrip=0.,goal_groups=None):
    out=[]
    for j in range(n):
        for trial in range(100000):
            lo,hi=region(j);q=rng.uniform(lo,hi)
            if not geom.valid_points(q[None])[0]:continue
            if out and np.min(np.linalg.norm(np.asarray(out)-q,axis=1))<2*r+.05:continue
            if avoid is not None and np.linalg.norm(q-avoid[j])<mintrip:continue
            out.append(q);break
        else:raise ValueError(f'Placement exhausted for agent {j}/{n}; geometry/capacity gap, no radius relaxation')
    return np.asarray(out)

def build(type_id,level=0,widen=0.,crossing=None):
    v=LADDERS[type_id][level];rng=np.random.default_rng(np.random.SeedSequence([SEED,type_id,level,round(widen*1000)]))
    name=f'xsc_stb_t{type_id:02d}_l{level}_w{round(widen*1000):03d}';r=.1;bc=None;notes=[]
    n=10;hx=hy=1.5;rects=[];barrier=None;free=None
    if type_id==1:n=int(v);r=.05
    elif type_id==2:
        n=2*int(v)
        if crossing is None:
            decision=HERE/'SCENE2_RESOLUTION.json'
            if not decision.exists():raise ValueError('Type2 longitudinal-queue geometry needs the explicit radius/extent resolution; old transverse formation is invalid.')
            crossing=read(decision)
        r=float(crossing['radius']);hx=hy=float(crossing['half_extent'])
        name=f'xsc_stb_t02_v2_l{level}_w{round(widen*1000):03d}'
        notes.append('Corrected longitudinal queues; radius/extent from explicit geometry resolution. Earlier transverse formation is archived and excluded from formal selection.')
    elif type_id==3:n=int(v);hx=2.;hy=(.9+widen)/2
    elif type_id==4:n=int(v);r=.13;hx=hy=1.2
    elif type_id==5:
        n=4*int(v);hx=hy=1.6;w=(.6+widen)/2
        free=[(-.7,-.7,.7,.7),(-hx,-w,hx,w),(-w,-hy,w,hy)]
        rects=rect_complement((-hx,-hy,hx,hy),free)+[(-.25,-.25,.25,.25)]
    elif type_id==6:n=8;r=float(v);name='lit_swap8_obstacles'
    elif type_id==7:
        from scene_helpers import forest_layout
        rects,layout_seed=forest_layout(int(v))
        notes.append(f'0.3m forest blocks, pair/wall clearance>=0.3m; geometry-only seed {layout_seed}.')
    elif type_id in (8,9):
        n=2*int(v)if type_id==8 else int(v);hx=1.625;hy=1.2
        barrier=((0.,),(((0.,.3+widen),),),.05)
    elif type_id==10:
        n=2*int(v);hx=2.4;hy=1.2
        # The original Config checks staging against full slab thickness.
        # Represent the same long corridor using rectangles and a dummy gate.
        rects=[(-.8,-hy,.8,-(.3+widen)/2),(-.8,(.3+widen)/2,.8,hy)]
    elif type_id==11:
        n=2*int(v);hx=2.8;hy=1.2;w=(.3+widen)/2
        rects=[(-1.2,-hy,1.2,-w),(-1.2,w,-.25,hy),(.25,w,1.2,hy),(-.25,w+.3,.25,hy)]
    elif type_id==12:
        n=2*int(v);hx=hy=2.2;w=(.5+widen)/2
        free=[(-1.6,-w,w,w),(-w,-w,w,1.6),(-2.2,-.6,-1.,.6),(-.6,1.,.6,2.2)]
        rects=rect_complement((-hx,-hy,hx,hy),free)
    elif type_id==13:
        n=3*int(v);hx=hy=1.7;w=(.6+widen)/2
        free=[(-hx,-w,hx,w),(-w,-hy,w,w)]
        rects=rect_complement((-hx,-hy,hx,hy),free)
    elif type_id==14:
        n=4;r=.05;hx=hy=1.6;w=(float(v)+widen)/2
        rects=[(w,w,hx,hy),(-hx,w,-w,hy),(-hx,-hy,-w,-w),(w,-hy,hx,-w)]
    elif type_id==15:
        n=int(v);hx=hy=1.6;w=(.4+widen)/2;th=.05
        for lo,hi in [(-1.6,-.8-w),(-.8+w,-th/2),(th/2,.8-w),(.8+w,1.6)]:
            if hi>lo:rects.append((-th/2,lo,th/2,hi));rects.append((lo,-th/2,hi,th/2))
    elif type_id==16:
        n=int(v);w=.4+widen;hx=1.2+2*w;hy=1.2
        rects=[(x-.4,y-.15,x+.4,y+.15)for x in (-(.8+w),0.,.8+w)for y in (-(.15+w/2),.15+w/2)]
        notes.append('Shelf gaps and both x-end aisles equal prescribed width; outer y staging regions follow fixed half-height1.2m.')
    elif type_id==17:n=int(v)
    if type_id not in (1,2,6,17):bc,geom=bcgeom(n,r,hx,hy,rects,barrier)
    else:
        cfg=RN.LiteratureConfig(name,n,r,.05,.5,1000,wall_radius=4/600)
        geom=GeometryRevision(cfg)
    rg=route_geometry(geom);states=[];routes=[]
    def uniform_region(j):return np.array([-hx,-hy]),np.array([hx,hy])
    for si in range(12):
        for attempt in range(200):
            if type_id in (1,6):
                R=.125*n if type_id==1 else 1.4
                angle=rng.uniform(-.035,.035)+np.arange(n)*2*np.pi/n
                goals=-R*np.stack([np.cos(angle),np.sin(angle)],axis=1)
                p=-goals+rng.uniform(-.045,.045,(n,2))
            elif type_id==2:
                k=n//2;spacing=2.5*(2*r);longitudinal=-3*r-spacing*np.arange(k)
                p=np.vstack([np.c_[longitudinal,np.zeros(k)],np.c_[np.zeros(k),longitudinal]])+rng.uniform(-.05,.05,(n,2))
                goals=-p
                if np.any(np.abs(p)>np.array([hx,hy])-r+1e-12):continue
            elif type_id==3:
                x=np.linspace(-1.7,-.35,n//2);p=np.vstack([np.c_[x,np.full(n//2,-.22)],np.c_[x,np.full(n//2,.22)]])+rng.uniform(-.02,.02,(n,2))
                goals=np.c_[-p[:,0],-p[:,1]]
            elif type_id in (5,13,14):
                arms=4 if type_id!=13 else 3;k=n//arms
                dirs=np.array([[1.,0.],[0.,-1.],[-1.,0.],[0.,1.]])[:arms]
                rr=np.linspace(.85,1.45,k)if k>1 else [1.]
                p=np.asarray([d*s for d in dirs for s in rr])+rng.uniform(-.018,.018,(n,2))
                if type_id==13:
                    # Each arm sends agents into the other two arms when k>=2.
                    dest=[(a+1+(j%2))%3 for a in range(arms)for j in range(k)]
                    slots={a:[]for a in range(arms)}
                    for idx,a in enumerate(dest):slots[a].append(idx)
                    goals=np.zeros_like(p)
                    for a,ids in slots.items():
                        for ix,s in zip(ids,np.linspace(.85,1.45,len(ids))):goals[ix]=dirs[a]*s
                else:goals=-p
            elif type_id in (8,9,10,11):
                neck=.025 if type_id in (8,9)else .8 if type_id==10 else 1.2
                def room(j,right=False):
                    if type_id==9:left=not right
                    else:left=(j<n//2) != right
                    return (np.array([-hx,-hy]),np.array([-neck-.15,hy])) if left else (np.array([neck+.15,-hy]),np.array([hx,hy]))
                p=sample(rng,geom,n,r,room)
                if type_id==9:goals=sample(rng,geom,n,r,lambda j:room(j,True))
                else:goals=p*np.array([-1,1])
            elif type_id==12:
                def room(j):return (np.array([-2.2,-.6]),np.array([-1.,.6]))if j<n//2 else(np.array([-.6,1.]),np.array([.6,2.2]))
                p=sample(rng,geom,n,r,room)
                goals=np.empty_like(p);goals[:n//2]=np.c_[-p[:n//2,1],-p[:n//2,0]];goals[n//2:]=np.c_[-p[n//2:,1],-p[n//2:,0]]
            elif type_id==15:
                p=sample(rng,geom,n,r,uniform_region)
                def goalroom(j):
                    signs=np.sign(p[j]);target=-signs if j%2==0 else signs*np.array([-1,1])
                    lo=np.where(target<0,-1.6,.15);hi=np.where(target<0,-.15,1.6)
                    return lo,hi
                goals=sample(rng,geom,n,r,goalroom)
            elif type_id==17:
                phase=rng.uniform(0,2*np.pi)
                a=phase+np.arange(n)*2*np.pi/n+rng.uniform(-.05,.05,n)
                p=1.4*np.stack([np.cos(a),np.sin(a)],axis=1)
                rows,cols={12:(3,4),16:(4,4),20:(4,5)}[n]
                x=(np.arange(cols)-(cols-1)/2)*.44
                y=(np.arange(rows)-(rows-1)/2)*.44
                goals=np.array([[xx,yy]for yy in y for xx in x])[rng.permutation(n)]
            else:
                p=sample(rng,geom,n,r,uniform_region)
                goals=sample(rng,geom,n,r,uniform_region,avoid=p,mintrip=1.5 if type_id==7 else 1. if type_id==4 else 0.)
            if not(geom.valid_points(p).all()and geom.valid_points(goals).all()and separate(p,r)):continue
            if type_id!=17 and not separate(goals,r):continue
            paths=[rg.path(a,b)for a,b in zip(p,goals)]
            if any(path is None for path in paths):continue
            state=dict(scene=name,index=si,N=n,positions=p.tolist(),velocities=np.zeros_like(p).tolist(),goals=goals.tolist())
            if bc:state['bottleneck_config']=bc
            states.append(state);routes.append(paths);break
        else:raise ValueError(f'{TITLES[type_id]} state {si}: failed finite valid state/path proposal budget')
    longest=max(float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())for paths in routes for path in paths)
    cfg=RN.LiteratureConfig(name,n,r,.05,.5,int(math.ceil(6*longest/.5/.05)),wall_radius=4/600).to_dict()
    enabled=any(not geom.visible(a,b)for s in states for a,b in zip(np.asarray(s['positions']),np.asarray(s['goals'])))
    alias=name if type_id==2 else f'xsc_stb_t{type_id:02d}_l{level}_w{round(widen*1000):03d}'
    result=dict(name=alias,title=TITLES[type_id],type_id=type_id,level=level,knob=v,widen=widen,
        config=cfg,bottleneck_config=bc,walls=geom.walls.tolist(),states=states,route_guidance=enabled,
        factor=type_id,notes=notes,bounds=[-hx,-hy,hx,hy],longest_initial_route=longest,
        initial_paths=[[path.tolist()for path in paths]for paths in routes],geometry_valid=True)
    result['definition_sha256']=digest(result);return result

def geometry(scene):
    if scene['bottleneck_config']:return bottleneck_geometry(db.canonical(scene['bottleneck_config']))
    return GeometryRevision(RN.LiteratureConfig(**scene['config']))

def build_all():
    out=HERE/'scene_design';out.mkdir(exist_ok=True);summary=[]
    for t in range(1,18):
        for level in range(len(LADDERS[t])):
            path=out/f't{t:02d}_l{level}.json'
            if path.exists():value=read(path)
            else:
                try:value=dict(status='valid',scene=build(t,level))
                except Exception as exc:value=dict(status='geometry_gap',type_id=t,level=level,error=str(exc))
                dump(path,value)
            summary.append(dict(path=str(path),**{k:v for k,v in value.items()if k!='scene'}))
            print(t,level,value['status'],value.get('error',''),flush=True)
    dump(HERE/'SCENE_DESIGN_INDEX.json',summary)

if __name__=='__main__':build_all()
