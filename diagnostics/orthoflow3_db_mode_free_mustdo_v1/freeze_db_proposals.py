#!/usr/bin/env python3
"""Freeze DB K16 mode-free samples and previously DB-adapted W1 critic scores."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import jax,jax.numpy as jnp,numpy as np
from flax import serialization
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=Path(__file__).resolve().parent
AUD=ROOT/'diagnostics/orthoflow3_pipeline_dataset_evidence_audit_v1'
HARD=ROOT/'diagnostics/orthoflow3_db_generator_necessity_v1'
GEN=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
TRANS=ROOT/'diagnostics/orthoflow3_toy_q_db_transfer_v1'
BASE=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1'
RANK=ROOT/'diagnostics/orthoflow3_ranking_aware_critic_v1'
sys.path.insert(0,str(ROOT))
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.train_generator import Generator,dist_params,CENTER,RADIUS
from shared_rollout_db.src.rollout_db import eta_identity
spec=importlib.util.spec_from_file_location('frozen_ranklib',RANK/'run_experiment.py')
lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    HERE.mkdir(exist_ok=True)
    freeze=json.loads((AUD/'db_hard_frozen_generator_mean.json').read_text())
    selected=json.loads((GEN/'generator_training.json').read_text())['selected']
    gc=Path(selected['checkpoint']);assert sha(gc)==freeze['generator_checkpoint_sha256']
    gm=Generator();gp=serialization.from_bytes(gm.init(jax.random.PRNGKey(0),jnp.zeros((1,80),jnp.float32),'DB'),gc.read_bytes())
    nm=json.loads((BASE/'dataset_manifest.json').read_text())
    sn=nm['state_normalization']['DB'];sc=np.asarray(sn['mean'],np.float32);ss=np.asarray(sn['std'],np.float32)
    ec=np.asarray(nm['eta_normalization']['center'],np.float32);es=np.asarray(nm['eta_normalization']['scale'],np.float32)
    hs=[];samples=[];states=[]
    for row in freeze['states']:
        h=np.asarray(json.loads((HARD/'raw'/f"conditioning_{row['state_id']}.json").read_text())['h_feature'],np.float32)
        assert h.shape==(80,)
        raw=gm.apply(gp,jnp.asarray(((h-sc)/ss)[None]),'DB')
        mu,sig=map(np.asarray,dist_params(raw));mu=mu[0];sig=sig[0]
        seed=int(hashlib.sha256(f"mode_free_db_hard_v1|seed{selected['seed']}|{row['state_id']}".encode()).hexdigest()[:16],16)
        eps=np.random.default_rng(seed).standard_normal((16,3))
        arr=np.asarray(CENTER)+np.asarray(RADIUS)*np.tanh(mu+sig*eps)
        hs.append((h-sc)/ss);samples.append(arr)
        states.append({'state_id':row['state_id'],'state_uid':row['state_uid'],'source_group':row['source_group'],
                       'eta':[list(map(float,x)) for x in arr],'mu_latent':mu.tolist(),'sigma_latent':sig.tolist(),
                       'seed_uint64':seed})
    hs=np.asarray(hs,np.float32);samples=np.asarray(samples,np.float32)
    cm=lib.SingleCritic();logits=[];cps=[]
    for seed in (17,23,41):
        cp=TRANS/'toy_full_finetune'/f'seed{seed}'/'checkpoint.msgpack'
        p=serialization.from_bytes(cm.init(jax.random.PRNGKey(seed),jnp.zeros((1,80)),jnp.zeros((1,3))),cp.read_bytes())
        flat_h=np.repeat(hs,16,axis=0);flat_eta=(samples.reshape(-1,3)-ec)/es
        logits.append(np.asarray(cm.apply(p,jnp.asarray(flat_h),jnp.asarray(flat_eta))).reshape(48,16))
        cps.append({'path':str(cp),'sha256':sha(cp),'seed':seed})
    score=np.mean(np.stack(logits),axis=0)
    for i,row in enumerate(states):row['critic_scores']=score[i].tolist();row['critic_choice_K4']=int(np.argmax(score[i,:4]));row['critic_choice_K16']=int(np.argmax(score[i]))
    manifest={'states':states,'generator_checkpoint':str(gc),'generator_sha256':sha(gc),
              'critic':'DB-adapted Toy W1, full-finetuned on DB TRAIN; three-seed logit ensemble',
              'critic_checkpoints':cps,'selection_before_proposal_outcomes':True,
              'proposal_seed_rule':'sha256(mode_free_db_hard_v1|selected_gen_seed|state_id)',
              'mode_ids_used':False,'manual_mode_correspondence_used':False}
    (HERE/'db_frozen_proposals.json').write_text(json.dumps(manifest,indent=2)+'\n')
    controller=freeze['hard_panel_controller_uid']
    seeds=[json.dumps({'future_index':i},sort_keys=True,separators=(',',':')) for i in range(16)]
    req=[{'state_uid':s['state_uid'],'eta_uid':eta_identity(s['eta'][j])[0],
          'controller_uid':controller,'seed_keys':seeds} for s in states for j in range(16)]
    (HERE/'db_proposal_planned_rollouts.json').write_text(json.dumps({'requests':req},indent=2)+'\n')
    print(json.dumps({'states':len(states),'proposals':len(req),'requested':len(req)*16,'critic':manifest['critic']}))
if __name__=='__main__':main()
