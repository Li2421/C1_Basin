def evaluate(p, g, safe, nominal, fields3, filters, walls, cfg, role, step=0,
             names=NEW, t5_seed='tangent', clearance_side=1, route_remaining=None):
    route_remaining=np.linalg.norm(np.asarray(g)-np.asarray(p),axis=1) if route_remaining is None else np.asarray(route_remaining)
    start=time.perf_counter();p=np.asarray(p,float);g=np.asarray(g,float)
    speed=cfg.max_speed;n=len(p);zero=np.zeros_like(p)
    fields={k:np.asarray(fields3[:,j])for j,k in enumerate(('G','P','R'))}
    requested=set(names);tau=.05*speed;safe_d=2*cfg.agent_radius+cfg.agent_collision_margin+1e-4
    meta=dict(svd_seconds=0.,svd_ranks=[],near_tolerance=tau,unavailable={},degenerate={},role=None)
    rows=[];near_pairs={};neighbors=[[]for _ in p]
    for i,f in enumerate(filters):
        ai=[]
        for k,(row,low,label) in enumerate(zip(f.A,f.lower,f.kinds)):
            norm=np.linalg.norm(row)
            if norm<1e-14:continue
            slack=float((row@safe[i]-low)/norm)
            ai.append((k,np.asarray(row)/norm,slack,label))
            if label[0]=='pair':
                j=int(label[2]);neighbors[i].append((slack,float(np.linalg.norm(p[i]-p[j])),j))
                if slack<=tau and not(route_remaining[i]<=cfg.goal_tolerance and route_remaining[j]<=cfg.goal_tolerance):
                    pair=tuple(sorted((i,j)));near_pairs[pair]=min(slack,near_pairs.get(pair,float('inf')))
        rows.append(ai)
    # One pair, selected only from current constraints; no summation of pairs.
    if role.pair is None and near_pairs:
        pair=min(near_pairs,key=lambda ij:(near_pairs[ij],ij))
        remaining=route_remaining
        unfinished=[i for i in pair if remaining[i]>cfg.goal_tolerance]
        passer=min(unfinished,key=lambda i:(remaining[i],i))
        role.pair=pair;role.passer=passer;role.yielder=next(i for i in pair if i!=passer);role.activation_step=step
    role_active=role.pair is not None and np.linalg.norm(p[role.pair[0]]-p[role.pair[1]])<=.5 and any(route_remaining[i]>cfg.goal_tolerance for i in role.pair)
    meta['role']=dict(pair=role.pair,yielder=role.yielder,passer=role.passer,activation_step=role.activation_step,active=role_active)
    for name in NEW:
        if name not in requested:continue
        out=zero.copy()
        if name=='T1':out=rotate(fields['G'])
        elif name=='T2':
            for ids in conflict_components(p,1.25*safe_d):
                q=p[ids]-p[ids].mean(axis=0);scale=speed/max(speed,float(np.linalg.norm(q,axis=1).max()))
                out[ids]=scale*rotate(q)
        elif name=='T3' and role_active:
            i,j=role.pair;q=rotate(p[i]-p[j]);q*=min(1.,speed/max(np.linalg.norm(q),1e-30));out[i]=q;out[j]=-q
        elif name=='T4':
            for i,rr in enumerate(rows):
                wall=[r for r in rr if r[3][0]=='wall' and r[2]<=tau]
                if wall:out[i]=speed*rotate(min(wall,key=lambda r:(r[2],r[0]))[1])
        elif name=='T5':
            d={'goal':fields['G'],'tangent':rotate(fields['G']),'nominal':nominal}[t5_seed]
            t0=time.perf_counter()
            for i,rr in enumerate(rows):
                active=[r[1]for r in rr if r[2]<=tau]
                # Include the first-order speed-ball normal at a near-active safe input.
                sn=np.linalg.norm(safe[i])
                if sn>=speed-tau and sn>1e-14:active.append(safe[i]/sn)
                if active:
                    a=np.asarray(active);_,s,vt=np.linalg.svd(a,full_matrices=False)
                    rank=int(np.sum(s>max(1e-12,1e-10*s[0])));v=vt[:rank]
                    out[i]=d[i]-v.T@(v@d[i])
                else:rank=0;out[i]=d[i]
                meta['svd_ranks'].append(rank)
            meta['svd_seconds']=time.perf_counter()-t0
        elif name=='P2':out=OrthoFlow3BasisFamily.raw_ortho_flow(fields['G'],nominal,speed)
        elif name=='R2':
            for i,near in enumerate(neighbors):
                relevant=[(s,j)for s,_,j in near if s<=3*tau]
                if relevant:
                    w=np.exp(-np.maximum([s for s,j in relevant],0)/tau)
                    out[i]=sum(v*(p[i]-p[j])for v,(s,j)in zip(w,relevant))/(w.sum()+1e-12)
        elif name=='R3':
            for i,near in enumerate(neighbors):
                eligible=[r for r in near if r[0]<=3*tau]
                if eligible:out[i]=p[i]-p[min(eligible)[2]]
        elif name in ('Y1','Y2','Y3','Y4','S1') and role_active:
            y,pa=role.yielder,role.passer;pg=g[pa]-p[pa];pn=np.linalg.norm(pg)
            if name=='Y1':out[y]=-fields['G'][y]
            elif name=='Y4':out[y]=-safe[y]
            elif name=='S1':out[y]=-fields['G'][y];out[pa]=fields['G'][pa]
            elif pn>1e-12:
                if name=='Y2':out[y]=speed*rotate(pg/pn)
                else:
                    foot=p[pa]+np.clip((p[y]-p[pa])@pg/(pn*pn),0.,1.)*pg
                    away=p[y]-foot;an=np.linalg.norm(away)
                    out[y]=speed*away/an if an>1e-10 else clearance_side*speed*rotate(pg/pn)
                    meta['degenerate']['Y3_collinear']=bool(an<=1e-10)
            else:meta['degenerate'][name+'_passer_goal']=True
            if name in ('Y1','S1') and np.linalg.norm(fields['G'][y])<1e-12:meta['degenerate'][name+'_zero_yielder_goal']=True
        elif name=='C1':
            # Only nearby finite straight boundaries with interior feet qualify.
            for i,rr in enumerate(rows):
                eligible=[]
                for r in rr:
                    if r[3][0]!='wall':continue
                    k=int(r[3][2]);a,b=np.asarray(walls[k]);v=b-a;vv=v@v
                    if vv<1e-20:continue
                    h=(p[i]-a)@v/vv
                    if not 1e-8<h<1-1e-8:continue
                    normal=r[1];dist=np.linalg.norm(p[i]-(a+h*v))
                    eligible.append((k,normal,dist,v/np.sqrt(vv)))
                choices=[]
                for aa in range(len(eligible)):
                    for bb in range(aa+1,len(eligible)):
                        x,z=eligible[aa],eligible[bb]
                        if x[1]@z[1]<-np.cos(np.pi/12) and abs(x[3]@z[3])>np.cos(np.pi/12):choices.append((x[2]+z[2],x[0],z[0],x[3]))
                if choices:
                    axis=min(choices,key=lambda q:q[:3])[3].copy()
                    if axis@fields['G'][i]<0:axis=-axis
                    elif abs(axis@fields['G'][i])<1e-14 and tuple(axis)<tuple(-axis):axis=-axis
                    out[i]=speed*axis
        elif name=='C2':
            # D=sum_i -tau_d log sum_k exp(-clearance_ik/tau_d).
            # Pair terms occur in both local softmins. Differentiate both ends.
            tau_d=.25*safe_d;D=0.;terms_count=0
            for i,rr in enumerate(rows):
                terms=[]
                for r in rr:
                    k,normal,slack,label=r
                    if label[0]=='pair':
                        j=int(label[2]);dist=np.linalg.norm(p[i]-p[j]);terms.append((dist-safe_d,normal,j))
                    elif label[0]=='wall':
                        a,b=np.asarray(walls[int(label[2])]);v=b-a
                        foot=a+np.clip((p[i]-a)@v/max(v@v,1e-30),0,1)*v
                        gap=np.linalg.norm(p[i]-foot)-cfg.agent_radius-cfg.wall_radius-cfg.wall_collision_margin-1e-4
                        terms.append((gap,normal,None))
                if terms:
                    gaps=np.asarray([x[0]for x in terms]);m=gaps.min();e=np.exp(-(gaps-m)/tau_d);w=e/e.sum();D+=m-tau_d*np.log(e.sum());terms_count+=len(terms)
                    for weight,(_,normal,j)in zip(w,terms):
                        out[i]+=weight*normal
                        if j is not None:out[j]-=weight*normal
            out*=speed;meta['clearance']=dict(D=float(D),tau=tau_d,terms=terms_count)
        # G and the common T2 scale already impose the radial bound; avoid a
        # second floating-point clipping pass, preserving exact historical aliases.
        fields[name]=out if name in ('T1','T2')else bounded_rows(out,speed)
        meta['unavailable'][name]=bool(np.linalg.norm(fields[name])<1e-10)
    meta['seconds']=time.perf_counter()-start
    return fields,meta
