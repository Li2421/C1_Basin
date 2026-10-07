"""Read-only synthesis of frozen factorial evidence and transfer prerequisites."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'orthoflow3_tt_ff_matched_learnability_v1'
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    f=FIELD/'family_study_v2';a=read(f/'analysis.json');decision=read(f/'decision.json');table=list(csv.DictReader((f/'factorial_summary.csv').open()))
    assert sha(f/'protocol.json')==read(f/'postflight.json')['protocol_sha256']
    for row in table:
        states=[s for s in a['states'] if row['scene']=='all' or s['scenario']==row['scene']]
        assert sum(s['cells'][row['chain']]['B15_count'][0] for s in states)==int(row['B15_lower'])
        assert sum(s['cells'][row['chain']]['B15_count'][1] for s in states)==int(row['B15_upper'])
    p=read(OLD/'protocol.json');t=np.load(OLD/'test_truth.npz');o=t['outcomes'];indices=t['indices'];rng=np.random.default_rng(817);direct=[]
    for scene in ('toy_give_way','ring_exchange'):
        ix=[j for j,i in enumerate(indices) if p['states'][int(i)]['scenario']==scene];b=o[:,ix]
        valid=(b[0,...,1]==0)&(b[1,...,1]==0);q=(b[...,0]*valid[None]).sum(-1)/valid.sum(-1)[None]
        robust=((b[...,0]==1)&(b[...,1]==0)).sum(-1)>=15
        dq=(q[1]-q[0]).mean(-1);dr=(robust[1].astype(float)-robust[0]).mean(-1);draw=rng.integers(0,len(ix),(20000,len(ix)))
        direct.append(dict(scene=scene,states=len(ix),Q_TT_FF=q.mean((1,2)).tolist(),Q_gain=float(dq.mean()),Q_gain_state_bootstrap95=np.quantile(dq[draw].mean(1),[.025,.975]).tolist(),
            B15_TT_FF=robust.sum((1,2)).tolist(),B15_gain=float(dr.mean()),B15_gain_state_bootstrap95=np.quantile(dr[draw].mean(1),[.025,.975]).tolist(),
            rescue=int((robust[1]&~robust[0]).sum()),break_count=int((robust[0]&~robust[1]).sum()),paired_units='state clusters; eta and seed records are not independent states'))
    write(ROOT/'paper_evidence.json',dict(factorial_table=table,factorial_effects=decision['state_paired_effects'],matched_confirmation=direct,
        checks='Factorial counts recomputed from per-state analysis; hashes checked; matched direct effects recomputed from seed truth',
        supported='Relocating correction into Flow improves sampled robust-eta prevalence under either safety placement in frozen Toy/Ring replication; not solely a safety-placement effect.',
        mechanism='Changed transport/effective correction response, not demonstrated expansion of attainable-action manifold; action diversity actually decreases.',
        limitations=['FT has mandatory terminal safety closure; not perfect factorial orthogonality',
          'TT includes native terminal decoding; eta0 negative control and integration refinement constrain but do not prove absence of all confounds',
          'Early generator-proposal pilot Toy robust count fell40 to26; empirical advantage is distribution/state dependent',
          '11 TT-only vs62 FF-only points: not set inclusion','No claim of continuous basin volume/topology or universal/LOSO dominance'],
        sources={str(path):sha(path) for path in (f/'analysis.json',f/'decision.json',f/'factorial_summary.csv',OLD/'test_truth.npz',FIELD/'validation_stage2/REPORT.md')}))
    rows=[
      dict(method='Controller-specific labels/context + partial counts',TT_evidence='Dropping negatives was a supervision bug; old controller context can misrepresent changed closed loop',FF_action='Already retained: actual FF response, independent UID, observed counts; never import TT labels',status='TRANSFERRED_DATA_SEMANTICS'),
      dict(method='Remove redundant independent h branch',TT_evidence='h shuffle/drop had little effect; response context already state-conditioned',FF_action='6 matched fits: same network/init/order/budget, measured h channels zeroed; masks retained',status='SMALL_RETRAIN'),
      dict(method='Strong eta prior + conditional correction',TT_evidence='eta prior strong; unrestricted Full sometimes damages selection; not a historically proven fix',FF_action='VAL-only probability shrinkage alpha0/.25/.5/.75/1; alpha0 is fallback, not interaction success',status='SMALL_INFERENCE_CONTROL'),
      dict(method='Multi-seed probability ensemble',TT_evidence='Historical hard Toy181→183 only; systematic errors remain',FF_action='Reuse3 existing/new seeds; secondary ensemble, no additional training',status='FREE_SECONDARY'),
      dict(method='W1 pair equality',TT_evidence='Helped combined wide Toy data, not structured-only universally',FF_action='Toy512/512 TRAIN pairs have n4, exactly pair-equal. Ring484/512 n4,24 n3,4 n2: near equal but not identical; retain observed-count likelihood instead of overweighting solver-missing pairs',status='TOY_EQUIVALENT_RING_NEAR_EQUAL_NOT_TESTED'),
      dict(method='Historical wide/off-anchor coverage',TT_evidence='Supported Toy continuous-eta prediction, dataset dependent',FF_action='Mechanism transferable, labels not; requires compatible FF wide data, deferred to user review',status='NOT_TRANSFER_OLD_LABELS'),
      dict(method='Rank loss / LCB / H80 / more capacity',TT_evidence='No stable selection gain; several harmful controls',FF_action='Not repeated in this bounded study',status='DEFERRED_NO_POSITIVE_TRANSFER_EVIDENCE')]
    with (ROOT/'transfer_audit.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(factorial_verified=True,matched_direct=direct,new_rollouts=0)),flush=True)

if __name__=='__main__':main()
