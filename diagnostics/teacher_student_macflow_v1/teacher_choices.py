"""DEVELOPMENT teacher reference from historical R5 and frozen round16 models."""
from core import *
from audit_utils import source_inventory
from diagnostics.ring_controller_critic_v1 import fit as old
import jax
jax.config.update('jax_enable_x64',False)
def main():
    names=read(D/'decentralized_pipeline_v1/main_protocol.json')['controller_order'];source=D/'controller_flow_blend_v1/stage_2/probes/R5_FEATURES.npz';z=np.load(source);oldnames=z['controller_names'].tolist();r5=z['features'][[oldnames.index(n) for n in names]];physical=old.physical23(read(HERE/'EVAL_STATES.json')).astype(np.float32);transform=D/'ring_controller_critic_v1/fits/full_transform.npz';tr={k:v for k,v in np.load(transform).items()};steps=read(D/'ring_controller_critic_v1/fits/selected_steps.json');arms=['F10','F13','S300','S500'];idx=[old.ARMS.index(a) for a in arms];choices={n:{} for n in names};files=[source,transform]
    for model in ['A0','A3']:
        values=[]
        for seed in old.SEEDS:
            ck=D/'ring_controller_critic_v1/fits/full'/model/f'seed{seed}'/f'step{steps[model]}.msgpack';files.append(ck);values.append(old.predict(model,seed,ck,tr,physical,r5))
        means=np.asarray(values)[:,:,:,idx].mean((0,2))
        for i,n in enumerate(names):choices[n][model]=arms[int(means[i].argmax())]
    write(HERE/'DEVELOPMENT_TEACHER_CHOICES.json',dict(utc=utc(),controllers=choices,evidence='DEVELOPMENT only; teacher labels opened previously; no student choices changed',descriptor_source=str(source),descriptor_pipeline='historical original-joint-flow R5; not new D-local probes',source_hashes={str(f):sha(f) for f in files},task_outcomes_used_as_model_input=False,new_rollouts=0,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory('imports_teacher_choices.json')
if __name__=='__main__':main()
