"""Zero-rollout canonical JSON and UID checks against actual previous main requests."""
from common import *
from requests_corrected import from_original,execution_entry
from replay import iter_array

OLD=ROOT/'diagnostics/cross_scene_correction_v1'

def main():
    plan={n:[('eta13','cross_entropy'),('sE05','prioritized_astar')]for n in ('lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_swap8_obstacles')}
    plan.update(xsc_swap_n12=[('eta13','prioritized_astar')],xsc_double_n4=[('eta13','prioritized_astar')])
    pending={(n,a,t)for n,items in plan.items()for a,t in items};cases=[];path=OLD/'main_CASES.json';digest=evidence(path)
    for c in iter_array(path):
        key=(c['state']['scene'],c['requested_action'],c['controller_payload']['nominal']['type_code'])
        if key not in pending:continue
        new=from_original(c,c['eta_a']);before=canonical(c).encode();after=canonical(new).encode()
        row=dict(scene=key[0],action=key[1],controller=c['controller_payload']['nominal']['controller'],state_index=c['state']['index'],
            original_rollout_uid=c['rollout_uid'],reconstructed_rollout_uid=new['rollout_uid'],
            canonical_json_bytes_equal=before==after,uid_equal=c['rollout_uid']==new['rollout_uid'],
            original_sha256=hashlib.sha256(before).hexdigest(),reconstructed_sha256=hashlib.sha256(after).hexdigest(),
            version=new['version'],payload_version=new['scenario_payload']['version'],execution_entry=execution_entry(new),
            scene_revision=new.get('scene_revision'),all_request_fields_preserved=set(c)==set(new))
        cases.append(row);pending.remove(key)
        assert row['canonical_json_bytes_equal']and row['uid_equal'],row
        write(HERE/'request_reconstruction'/f"{new['rollout_uid']}.json",dict(original=c,reconstructed=new))
        if not pending:break
    assert not pending,pending
    proof=ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ceiling_worker.py'
    circle=ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ceiling_circle.py'
    # Read-only source proof: the historical circle search dispatch and our
    # isolated execution use exactly the same old_run numerical implementation.
    evidence(proof);evidence(circle)
    assert 'from runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022 import run as old_run'in proof.read_text()
    source=ROOT/'diagnostics/cross_scene_correction_v1/backend/runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022.py'
    local=HERE/'backend'/source.name;assert sha(source)==sha(local)
    write(HERE/'REQUEST_RECONSTRUCTION_CHECK.json',dict(utc=utc(),passed=True,cases=len(cases),checks=cases,
        source=str(path),source_sha256=digest,constructor=str(HERE/'requests_corrected.py'),constructor_sha256=sha(HERE/'requests_corrected.py'),
        canonical_json_and_uid_bitwise_equal=True,circle_search_entry_source=str(proof),circle_search_entry_sha256=sha(proof),
        circle_numerical_source_sha256=sha(local),new_rollouts=0,models_retrained=0))
    print('Canonical request reconstruction passed:',len(cases),'requests; zero rollouts.',flush=True)

if __name__=='__main__':main()
