"""Pilot only: immutable K4 regime map, no selection generalization claim."""
from common import *
import numpy as np
from shared_rollout_db.src.rollout_db import connect

def interval(x):
    s=(x==1).mean(-1);u=(x==-1).mean(-1)
    return s,u
def paired(a,b):
    # Same seed physical outcomes. Unknown cancels when comparing same eta;
    # callers below compare different eta, so conservative interval is [0,1].
    la,ua=interval(a);lb,ub=interval(b)
    return la-lb-ub,la+ua-lb

def main():
    folder=HERE/'stage_1';p=read(folder/'protocol.json');post=read(folder/'POSTFLIGHT.json');assert post['missing']==post['quarantined']==0
    names=p['controller_order'];states=p['states'];etas=p['eta_indices'];x=np.full((12,24,4,4),-2,dtype=np.int8)
    with connect(True) as db:
        for r in read(folder/'CASES.json'):
            z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone();assert z and not z['conflict_quarantined']
            x[names.index(r['controller_name']),r['state_index'],etas.index(r['eta_index']),r['seed']]=-1 if z['numerical_failure'] else z['success']
    assert not (x==-2).any()
    np.savez_compressed(folder/'LABELS.npz',outcomes=x,controller_names=np.array(names),eta_indices=np.array(etas),state_uids=np.array([s['state_uid'] for s in states]))
    rng=np.random.default_rng(2026101303);draws=rng.integers(24,size=(5000,24));rows=[]
    for c,name in enumerate(names):
        lower,unknown=interval(x[c]);delta,upper=paired(x[c,:,etas.index(13)],x[c,:,etas.index(10)])
        observed=np.divide((x[c]==1).sum((0,2)),((x[c]==0)|(x[c]==1)).sum((0,2)),out=np.full(4,np.nan),where=((x[c]==0)|(x[c]==1)).sum((0,2))>0)
        assert np.isfinite(observed).all()
        q=p['controllers']['ring_exchange'][name]['payload']['blend']
        winner=13 if delta.mean()>0 else 10 if upper.mean()<0 else None
        rows.append(dict(controller=name,pair=q['pair'],A=q['A'],B=q['B'],lambda_value=q['lambda'],eta_indices=etas,success_lower=lower.mean(0).tolist(),success_upper=(lower+unknown).mean(0).tolist(),observed_valid_success=observed.tolist(),
            regime_K4=etas[int(observed.argmax())],eta13_minus_eta10_lower=float(delta.mean()),eta13_minus_eta10_upper=float(upper.mean()),ci95_conservative_delta=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
            unambiguous_winner_13_vs_10=winner,winning_eta_success_lower=None if winner is None else float(lower[:,etas.index(winner)].mean()),unknown=int((x[c]==-1).sum())))
    pairs=[]
    for i in range(4):
        rr=[r for r in rows if r['pair']==i];good13=[r['lambda_value'] for r in rr if r['unambiguous_winner_13_vs_10']==13 and r['winning_eta_success_lower']>=.4];good10=[r['lambda_value'] for r in rr if r['unambiguous_winner_13_vs_10']==10 and r['winning_eta_success_lower']>=.4]
        pairs.append(dict(pair=i,A=rr[0]['A'],B=rr[0]['B'],eta13_competent_lambdas=good13,eta10_competent_lambdas=good10,passes=bool(good13 and good10)))
    # Endpoints are incomplete K4 labels: show exact coverage, never backfill.
    prev=read(PREVIOUS/'protocol.json');old=np.load(PREVIOUS/'LABELS.npz')['outcomes'][:,:24,:,:4];anchors=[]
    for name in dict.fromkeys([r['A'] for r in pairs]+[r['B'] for r in pairs]):
        mat=old[prev['controller_order'].index(name)];ar=[]
        for eta in etas:
            mask=(mat[:,eta]!=-2).any(-1);v=mat[mask,eta]
            ar.append(dict(eta=eta,states=int(mask.sum()),seed_slots=int(v.size),success_lower=float((v==1).mean()),success_upper=float(((v==1)|(v==-1)).mean()),unknown=int((v==-1).sum())))
        anchors.append(dict(controller=name,curve=ar))
    passed=sum(r['passes'] for r in pairs)>=3
    result=dict(status='COMPLETE',scope='DEVELOPMENT PILOT; locates internal regime boundary, not eta selection test',rows=rows,pairs=pairs,gate=dict(passed=passed,passing_pair_count=sum(r['passes'] for r in pairs),required_pairs=3,uses_internal_lambdas_only=True,next_stage='STAGE 2' if passed else 'NOT EXECUTED — GATE FAILED'),
        original_endpoint_partial_curves=anchors,raw_requests=4608,valid=post['valid'],unknown=post['unknown'],new_rollouts=read(folder/'CACHE_INVENTORY.json')['missing'],
        reused_endpoint_seed_slots=sum(r['seed_slots'] for a in anchors for r in a['curve']),runtime_sha256=p['runtime_sha256'],protocol_sha256=sha(folder/'protocol.json'),preregistration_sha256=p['preregistration_sha256'],bootstrap_draws=5000,bootstrap_seed=2026101303,timing=read(folder/'PILOT_TIMING.json'))
    write(folder/'results.json',result);write(HERE/'STAGE_1_GATE.json',result['gate']);print(json.dumps(result['gate']))

if __name__=='__main__':main()
