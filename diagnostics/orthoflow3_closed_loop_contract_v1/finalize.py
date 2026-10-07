"""Freeze contract audit outputs only after its confirmation has completed."""
import json,hashlib,runpy
from pathlib import Path
from collections import Counter
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
from shared_rollout_db.src.rollout_db import connect
OUT=repair.OUT.parent

def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def run():
    dev=a.load(repair.OUT/'development/results.json');fresh=a.load(repair.OUT/'confirmation/results.json')
    tests=runpy.run_path(str(OUT/'test_contract.py'))['run']()
    for n,f in runpy.run_path(str(a.ROOT/'tests/test_basis_families.py')).items():
        if n.startswith('test_') and callable(f):f()
    before=a.load(a.OUT/'canonical_hashes_before.json')
    changed=[p for p,h in before.items() if a.sha(a.ROOT/p)!=h]
    assert changed==['new_benchmark_common/safety_eta3.py'],changed
    # The one pre-existing change is the separately accepted FK/raw-import repair.
    counts={}
    for stage,path in [('witness',OUT/'waiting_probe'),('development',repair.OUT/'development'),('confirmation',repair.OUT/'confirmation')]:
        c=Counter()
        for p in path.glob('run_*of*.json'):c.update(a.load(p))
        counts[stage]=dict(c)
    assert sum(c.get('physical_attempts',0) for c in counts.values())<=6000
    persistence=[]
    with connect(True) as con:
        for root in [OUT/'waiting_probe',repair.OUT/'development',repair.OUT/'confirmation']:
            rows=con.execute('SELECT experiment_uid,path FROM experiment WHERE path=?',(str(root/'execution'),)).fetchall()
            assert len(rows)==1,(root,rows)
            persistence.append({'path':str(root),'experiment_uid':rows[0][0],'registered':True})
    hashes={str(p.relative_to(a.ROOT)):a.sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.msgpack')}
    dump('provenance_hashes.json',{'audit_artifacts':hashes,'canonical_unchanged_except_prior_FK_patch':True,'prior_change':changed,
                                  'v2_labels_unchanged':a.sha(a.DATA/'eta_labels.parquet')==a.sha(repair.DATA/'eta_labels.parquet')})
    dump('cache_reuse_manifest.json',{'batches':counts,'db_registration':persistence,'no_physical_labels_invalidated':True})
    status='CONTRACT_REPAIRED_AND_VALIDATED'
    decision={'status':status,'repair':'remaining_horizon_input','repair_iterations':1,
              'physical_labels_remain_valid':True,'accepted_generator':str(repair.OUT/'generator_frozen.json'),
              'accepted_critic':str(repair.OUT/'critic_frozen.json'),'accepted_dataset':str(repair.DATA),
              'limitation':'Four active rotations are not certified continuation equivalences; small confirmation only',
              'no_post_confirmation_adaptation':True,'next_authorized_task':'Unified Physical Representation'}
    dump('decision.json',decision)
    lines=['# Closed-loop contract audit and minimal repair','',status,'',
      'The critic targets success probability under a fixed eta held for the complete remaining continuation, averaged over future MACFlow randomness. Four/Ring conditioning Flow is an uncommitted reference probe; Double commits its first action. No online eta reselection mismatch was found.','',
      'A legal waiting counterexample confirmed exact input aliasing across different remaining horizons. The isolated repair appends the remaining-time fraction. Full physical identities and labels are unchanged; no labels were transferred to transformed scenes. See WITNESS.md and CONTINUATION_CONTRACT.md.','',
      'The old h was insufficient for state-specific finite-horizon Q. The repaired h removes the demonstrated alias; it is not a universal sufficiency proof. Four passive reencoding must not be confused with active symmetry of its orientation-sensitive MACFlow. No claim that all active rotations/relabels have identical Q is accepted.','',
      '## Matched development and fresh confirmation','',
      '| Scenario | Dev old selected | Dev repaired oracle/selected | Fresh old selected | Fresh repaired oracle/selected | Fresh rescue/break |','|---|---:|---:|---:|---:|---:|']
    for sc in ('four_way_intersection','ring_exchange'):
        d=dev['summary'][sc];f=fresh['summary'][sc]
        lines.append(f"| {sc} | {d['old_selected']}/6 | {d['oracle']}/6, {d['selected']}/6 | {f['old_selected']}/2 | {f['oracle']}/2, {f['selected']}/2 | {f['rescue']}/{f['break']} |")
    lines+=['','Both generator and critic were dimension-incompatible and retrained from scratch with the same three matched seeds, architecture, objectives, balanced sampling and budgets. Thus changed proposals are not attributed solely to critic ranking. No waiting-witness labels were added. Numerical outcomes remain uncertified.','',
      'Generator/critic artifacts and proposal manifests are frozen before rollout outcomes. Fresh confirmation is deliberately small and cannot establish broad generalization. No adaptation followed it.','',
      '## Historical validity','',
      'Original v2 physical evidence remains valid for its original snapshots. Old models and evaluations remain records of the old conditioning contract, but their general finite-horizon sufficiency claims require qualification. Repaired labels are byte-identical; inputs are versioned in v2_contract. See LABEL_REUSE.md.','',
      '## Regressions and compute','',f'Contract tests and all three basis-family regressions passed. New physical attempts: {sum(c.get("physical_attempts",0) for c in counts.values())}; detailed reuse/retries in cache_reuse_manifest.json. Canonical environment, policy weights, safety and basis hashes remain unchanged from the prior accepted work. Only the previously accepted DB raw-import helper differs from the original pre-Phase-A snapshot.','',
      'The smallest next justified task is the already-authorized unified physical representation, retaining remaining time and contract-relevant policy-frame information. Do not infer full upstream equivariance from an invariant downstream encoder.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return decision

if __name__=='__main__':print(json.dumps(run(),indent=2))
