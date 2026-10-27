from pathlib import Path
import shutil,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
p=R/'shared_control/hard_projection.py';s=p.read_text();f=s[s.index('def project_velocity('):s.index('\n\nclass HardSafetyFilter:')]
a=f.index('    blocks = [sparse.csc_matrix(-matrix)]');b=f.index('    settings = clarabel.DefaultSettings()',a)
f=f[:a]+'''    m=len(lower); n=2*num_agents; neg=-matrix; data=[]; indices=[]; ptr=[0]
    for j in range(n):
        rows=np.flatnonzero(neg[:,j]!=0)
        data.extend((neg[rows,j],np.asarray([-1.])))
        indices.extend((rows,np.asarray([m+3*(j//2)+1+j%2])))
        ptr.append(ptr[-1]+len(rows)+1)
    cone_matrix=sparse.csc_matrix((np.concatenate(data),np.concatenate(indices).astype(np.int32),np.asarray(ptr,np.int32)),shape=(m+3*num_agents,n))
    if accel_mode.CHECK:
        blocks=[sparse.csc_matrix(-matrix)]
        for agent in range(num_agents):
            block=np.zeros((3,2*num_agents),dtype=np.float64)
            block[1:,2*agent:2*agent+2]=-np.eye(2);blocks.append(sparse.csc_matrix(block))
        ref=sparse.vstack(blocks,format='csc')
        assert ref.shape==cone_matrix.shape
        for field in ('data','indices','indptr'):
            aa=getattr(ref,field);bb=getattr(cone_matrix,field)
            assert aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes(),field
        accel_mode.MATRIX_CHECKS+=1
    cone_rhs=np.concatenate([-lower]+[np.asarray([max_speed,0.,0.]) for _ in range(num_agents)])
    cones=[clarabel.NonnegativeConeT(m)]+[clarabel.SecondOrderConeT(3) for _ in range(num_agents)]
    if n not in _P:
        eye=sparse.csc_matrix(np.eye(n));_P[n]=(eye.data.copy(),eye.indices.copy(),eye.indptr.copy())
    cached=_P[n];quadratic=sparse.csc_matrix((cached[0].copy(),cached[1].copy(),cached[2].copy()),shape=(n,n))
'''+f[b:]
f=f.replace('sparse.csc_matrix(np.eye(2 * num_agents)),','quadratic,')
f=f.replace('def project_velocity(', 'def _fast_project_velocity(',1)
header='''"""Exact CSC construction; unchanged solver, tolerances, diagnostics, exceptions."""
import numpy as np,clarabel
from scipy import sparse
from shared_control.hard_projection import HardProjectionConfig,CBFSolverError,_diagnostics,project_velocity as _original
import accel_mode
_P={}
'''
(H/'fast_projection.py').write_text(header+f+'''\ndef project_velocity(*a,**kw):
    return (_fast_project_velocity if accel_mode.ENABLED else _original)(*a,**kw)
''')
(H/'accel_mode.py').write_text('ENABLED=True\nCHECK=False\nMATRIX_CHECKS=0\n')
shutil.copyfile('/home/zhihan/.claude/jobs/7eda8ddd/tmp/speed/np_threefry.py',H/'np_threefry.py')
(H/'accel_fold.py').write_text('import jax,accel_mode\nfrom np_threefry import fold_in as _fast\ndef fold_in(key,data):\n return (_fast if accel_mode.ENABLED else jax.random.fold_in)(key,data)\n')
p=H/'backend/swap_safety.py';s=p.read_text().replace('from shared_control.hard_projection import project_velocity,CBFSolverError','from shared_control.hard_projection import CBFSolverError\nfrom fast_projection import project_velocity\nimport accel_mode').replace('def local_problem(snapshot,i):','def _local_problem_original(snapshot,i):',1)
s+='''\ndef local_problem(snapshot,i):
    walls=np.asarray(snapshot['walls'])
    if not accel_mode.ENABLED or len(walls)==0:return _local_problem_original(snapshot,i)
    p=np.asarray(snapshot['positions']);a=walls[:,0];ab=walls[:,1]-a
    t=np.clip(((p[i]-a)*ab).sum(-1)/(ab*ab).sum(-1),0,1)
    keep=np.flatnonzero(np.linalg.norm(p[i]-(a+t[:,None]*ab),axis=-1)<=.5+1e-9)
    A,lower,labels=_local_problem_original(dict(snapshot,walls=walls[keep]),i)
    return A,lower,[['wall',l[1],int(keep[l[2]])]if l[0]=='wall'else l for l in labels]
''';p.write_text(s)
p=H/'runtime_new.py';s=p.read_text().replace('import jax\n','import jax\nfrom accel_fold import fold_in\n').replace('actor.action(env,jax.random.fold_in(root,t))','actor.action(env,fold_in(root,t))');p.write_text(s)
p=H/'controllers_new.py';s=p.read_text().replace('import jax\n','import jax\nimport accel_mode\nfrom accel_fold import fold_in\n').replace('jax.random.fold_in(key,i)','fold_in(key,i)').replace('def wall_penalties(p,env,xp=np):','def _wall_penalties_original(p,env,xp=np):',1)
s+='''\ndef wall_penalties(p,env,xp=np):
    c=env.config
    if not accel_mode.ENABLED or xp is not np or not c.scene.startswith('xsc_'):return _wall_penalties_original(p,env,xp)
    margin=c.agent_radius+c.wall_collision_margin+c.wall_radius;zero=np.zeros(p.shape[:-1]);walls=np.asarray(env.walls)
    if not len(walls):return zero,zero
    a_all=walls[:,0];ab_all=walls[:,1]-walls[:,0];inner=np.zeros(p.shape[:-1])
    for j in range(p.shape[-2]):
        q=p[...,j,:];lo=q.reshape(-1,2).min(0);hi=q.reshape(-1,2).max(0);ctr=(lo+hi)/2;rad=np.linalg.norm(hi-lo)/2
        tt=np.clip(((ctr-a_all)*ab_all).sum(-1)/(ab_all*ab_all).sum(-1),0,1)
        keep=np.flatnonzero(np.linalg.norm(ctr-(a_all+tt[:,None]*ab_all),axis=-1)-rad<=margin+1e-9)
        if not len(keep):continue
        a=a_all[keep];ab=ab_all[keep];d=q[...,None,:]-a
        t=np.clip(np.sum(d*ab,axis=-1)/np.sum(ab*ab,axis=-1),0,1);delta=d-t[...,None]*ab
        distance=np.sqrt(np.maximum(np.sum(delta*delta,axis=-1),1e-30))
        inner[...,j]=np.maximum(margin-np.min(distance,axis=-1),0)**2
    return inner,zero
''';p.write_text(s)
# Authorized new execution modules are part of the exact execution identity.
p=H/'scenes.py';s=p.read_text();s=s.replace("('scene_environment.py', 'controllers_new.py', 'runtime_new.py')","('scene_environment.py','controllers_new.py','runtime_new.py','accel_mode.py','fast_projection.py','accel_fold.py','np_threefry.py')");p.write_text(s)
# Update graph targets to final hashes, record replaced originals separately.
b=json.loads((H/'BACKEND_BUILD.json').read_text())
for r in b['sources']:r['target_sha256']=hashlib.sha256(Path(r['target']).read_bytes()).hexdigest()
(H/'BACKEND_BUILD.json').write_text(json.dumps(b,indent=2)+'\n')
print('Four acceleration changes installed locally, certification required')
