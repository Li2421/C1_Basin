"""Old frozen learning-only models on the same26-action development menus."""
from common import *
from inference import predict
from select_models import choose_sampled
from train_critic import load_data
import jax
def main():
    data=load_data();types=data['types'];old=D/'crosstype1c_two_eta_schedule_v1';frozen=read(old/'FROZEN_PIPELINE.json');out=np.full((len(types),48),-1,np.int32)
    with jax.default_matmul_precision('highest'):
        for held in sorted(set(types)):
            mask=types==held;group=frozen['outer'].get(str(held),frozen['all_types']);subset={k:v[mask] if k in ('names','types','features','action','labels','candidate_indices','uids','collision','timeout','unknown') else v for k,v in data.items()};p=predict(group['full'],subset);out[mask]=choose_sampled(p,data['candidate_indices'][mask])[frozen['primary_method']]
    with np.load(old/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:oldnames=list(z['names']);original=z[frozen['primary_method']]
    for j,n in enumerate(oldnames):assert np.array_equal(out[list(data['names']).index(n)],original[j]),('Old62 sampled-choice mismatch',n)
    np.savez_compressed(HERE/'LEGACY_DEVELOPMENT_CHOICES.npz',names=data['names'],types=types,legacy_pipeline=out)
    seal('LEGACY_DEVELOPMENT_CHOICES.receipt.json',[HERE/'LEGACY_DEVELOPMENT_CHOICES.npz',HERE/'legacy_development.py',old/'FROZEN_PIPELINE.json'],old62_sampled_choices_exact=True,new42_use_old_all9_model=True,development_only=True,new_evaluation_dispatch='All9 for every type per revised prompt')
if __name__=='__main__':main()
