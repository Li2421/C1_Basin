"""Recompute certificates from saved evidence; generate no rollout or training."""
from common import *
from single_integrator.environment import segment_distance,point_segment_distance

def main():
    catalog=read(HERE/'CATALOG_FINAL.json');summary={};proofs=[]
    for family,states in catalog['catalog'].items():
        minimum_wall=np.inf;minimum_pair=np.inf;maximum_speed=0.;source_count=0
        for s in states:
            ref=s['joint_proof'];raw=read(ref['raw_path']);assert raw['success']==1 and raw['collision']==0 and raw['numerical_failure']==0
            assert sha(raw['trace_path'])==ref['trace_sha256']==raw['trace_sha256']
            z=np.load(raw['trace_path']);p=z['positions'];a=z['actions'];walls=z['walls'];goals=z['goals']
            assert len(p)==len(a)+1 and np.array_equal(p[0],np.asarray(s['positions'])) and np.array_equal(goals,np.asarray(s['goals']))
            assert np.max(np.linalg.norm(goals-p[-1],axis=1))<=.08
            speed=float(np.linalg.norm(a,axis=-1).max());assert speed<=.5+1e-10
            before=p[:-1];after=p[1:];wall=float(segment_distance(before[:,:,None],after[:,:,None],walls[None,None,:,0],walls[None,None,:,1]).min()-(.16+4/600+.005))
            ii,jj=np.triu_indices(4,1);pair=float(point_segment_distance(np.zeros(2),before[:,ii]-before[:,jj],after[:,ii]-after[:,jj]).min()-.32)
            assert wall>0 and pair>0
            minimum_wall=min(minimum_wall,wall);minimum_pair=min(minimum_pair,pair);maximum_speed=max(maximum_speed,speed);source_count+=1
            proofs.append(dict(family=family,state_alias=s['alias'],raw_path=ref['raw_path'],trace_sha256=ref['trace_sha256'],steps=len(a),wall_clearance=wall,pair_clearance=pair,passed=True,reference_horizon=10000,evaluation_horizon=s['H']))
        training_seeds={s['geometry_seed'] for s in states if s['role']=='train'};eval_seeds={s['geometry_seed'] for s in states if s['role']=='evaluation'}
        assert len(training_seeds)==16 and len(eval_seeds)==8 and not training_seeds&eval_seeds
        provenance=read(HERE/'datasets'/f'{family}_PROVENANCE.json')
        assert all(r['source_geometry_seed'] in training_seeds and r['source_geometry_seed'] not in eval_seeds for r in provenance)
        effective_train={r['source_geometry_seed'] for r in provenance if r['success'] and r['split']=='train'}
        summary[family]=dict(valid_states=source_count,single_agent_paths=4*source_count,joint_successes=source_count,
            training_geometries=16,evaluation_geometries=8,effective_training_geometries=len(effective_train),
            minimum_swept_wall_clearance=minimum_wall,minimum_swept_pair_clearance=minimum_pair,max_speed=maximum_speed,
            heldout_geometry_seed_overlap=0,evaluation_geometry_in_policy_training=False)
    receipt=read(HERE/'PREREGISTRATION_HOLDOUT_CORRECTION.receipt.json')
    assert all(sha(p)==value for p,value in receipt['checkpoints'].items())
    for path in (HERE/'policies').glob('*/COMPLETE.receipt.json'):
        r=read(path);assert all(sha(p)==value for p,value in r['files'].items())
    for family in 'ABCDEFG':
        nav=read(HERE/f'NAVIGATION_VALID_CAPABILITY_{family}.json');cases=read(HERE/f'navigation_valid_{family}_CASES.json')
        assert len(cases)==64 and all(c['state']['geometry_seed'] in {s['geometry_seed'] for s in catalog['catalog'][family] if s['role']=='evaluation'} for c in cases)
        assert all(r['episodes']==16 for r in nav['instances'])
        assert not nav['family_passed']
    write(HERE/'EVIDENCE_VERIFICATION.json',dict(passed=True,summary=summary,proofs=proofs,
        policy_weights_unchanged_after_original_screen=True,training_and_evaluation_geometry_disjoint=True,
        capability_gate_uses_valid_holdout=True,new_rollouts_generated_by_verification=0))
    print('Saved evidence certificates verified',summary,flush=True)

if __name__=='__main__':main()
