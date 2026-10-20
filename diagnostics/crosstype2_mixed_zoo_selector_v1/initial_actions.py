"""No physics: eight independent initial-observation nominal-action queries."""
from common import *
from mixed_controllers import make
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
import jax
jax.config.update('jax_enable_x64',False)


def main(phase,index,workers):
    p=read(HERE/f'{phase}_protocol.json');cache={}
    for name in [n for n in p['controller_order'] if p['types'][n] not in ('Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo')][index::workers]:
        target=HERE/phase/'initial_actions'/f'{name}.npz'
        receipt=target.with_suffix('.receipt.json')
        if receipt.exists():
            assert sha(target)==read(receipt)['sha256'];continue
        rt=make(name,p,cache);actions=[]
        for s in p['states']:
            current=[]
            for seed in range(8):
                env=rt.make_env(s)
                _,action,_=rt.nominal(env,key_for(s,seed,0))
                assert env.step_count==0
                current.append(np.asarray(action).copy())
            actions.append(current)
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,actions=np.asarray(actions))
        write(receipt,dict(utc=utc(),sha256=sha(target),controller=name,physical_steps=0,terminal_labels_read=False,
              checkpoint_sha256=p['profiles'][name]['payload']['flow_checkpoint_sha256'],source_code_sha256=sha(__file__)))
        print('Initial nominal-action queries complete:',name,flush=True)

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
