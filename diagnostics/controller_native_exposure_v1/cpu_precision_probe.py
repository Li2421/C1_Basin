"""One 200-step integrity block on each allocated CPU type; no model selection."""
import os,sys,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
if len(sys.argv)==1:
    cpus=sorted(os.sched_getaffinity(0));assert cpus[0]<8<=cpus[-1],cpus
    for cpu in [cpus[0],cpus[-1]]:
        subprocess.run([sys.executable,'-m','diagnostics.controller_native_exposure_v1.cpu_precision_probe',str(cpu)],check=True)
else:
    cpu=int(sys.argv[1]);os.sched_setaffinity(0,{cpu})
    # Reuse setup without re-executing its former three-seed probe.
    namespace={};text=(here/'replay_probe.py').read_text();exec(text[:text.index('result=[]')],namespace)
    old,new=namespace['old'],namespace['new'];np=namespace['np'];jax=namespace['jax'];seed=17
    a=namespace['setup'](old,next(i for i,j in enumerate(old.JOBS) if j==dict(fold='joint',arm='union_native_slots',seed=seed)))
    d=jax.numpy.asarray(np.load(here/f'draws_seed{seed}.npz')['joint'][:200]);(p,o),loss=a['block'](a['params'],a['opt'],d)
    initial=namespace['serialization'].to_bytes(a['params']);final=namespace['serialization'].to_bytes(p)
    row=dict(cpu=cpu,initial_sha=namespace['hashlib'].sha256(initial).hexdigest(),final_sha=namespace['hashlib'].sha256(final).hexdigest(),mean_loss=float(np.mean(loss)))
    (here/f'cpu{cpu}_block200.msgpack').write_bytes(final);np.save(here/f'cpu{cpu}_loss200.npy',np.asarray(loss));new.write(here/f'cpu{cpu}_precision.json',row);print(json.dumps(row),flush=True)
