"""Create a non-frozen, reflection-equivariant Gap1 Flow sampling candidate.

The underlying weights are copied exactly. Only the optional Flow
reflection-averaging flag is set; hard safety, observations and environment
remain unchanged. This is for non-opposing competence screening only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import flax.core
import numpy as np

from new_benchmark_common.macflow import load_checkpoint, save_checkpoint


def make(dataset:Path,source:Path,output:Path,audit:Path):
    dataset,source,output,audit=map(Path,(dataset,source,output,audit))
    if output.exists():
        raise FileExistsError(output)
    manifest=json.loads((dataset/'manifest.json').read_text())
    n=int(manifest['scenario_config']['num_agents'])
    symmetry=json.loads(audit.read_text())
    if (symmetry['dataset_manifest_sha256']!=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
            or symmetry['exact_observation_reflection_max_error']!=0
            or symmetry['exact_expert_action_reflection_max_error']!=0):
        raise ValueError('paired DEV observation/target reflection audit failed')
    agent,metadata=load_checkpoint(source,
        expected_environment_fingerprint=manifest['environment_fingerprint'])
    if agent.config['num_agents']!=n or agent.config['obs_dim']!=8 or agent.config['act_dim']!=2:
        raise ValueError('candidate requires an eight-feature planar Gap Flow matching the dataset')
    om=np.asarray(agent.config['obs_mean']).reshape(n,8)
    am=np.asarray(agent.config['act_mean']).reshape(n,2)
    if np.max(np.abs(om[:,[0,2,4,6]]))>1e-8 or np.max(np.abs(am[:,0]))>1e-8:
        raise ValueError('checkpoint normalization does not commute with horizontal reflection')
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=dict(agent.config)
    cfg['reflect_average']=True
    candidate=agent.replace(config=flax.core.FrozenDict(cfg))
    meta=dict(metadata)
    meta.update(candidate_adapter='paired_x_reflection_average_of_two_standard_flow_integrations',
                source_checkpoint=str(source),source_checkpoint_sha256=source_sha,
                symmetry_audit=str(audit),selection_status='nonopposing_screen_only_not_frozen')
    output.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(output,candidate,meta)
    report=dict(source=str(source),source_sha256=source_sha,output=str(output),
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                symmetry_audit=str(audit),weights_unchanged=True,reflect_average=True,
                no_safety_change=True)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--symmetry-audit',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(make(args.dataset,args.source,args.output,args.symmetry_audit),indent=2))


if __name__=='__main__':
    main()
