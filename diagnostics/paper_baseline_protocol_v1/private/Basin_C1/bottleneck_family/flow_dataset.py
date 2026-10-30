"""Independent successful expert trajectories for joint Stage-I MACFlow."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from new_benchmark_common.dataset import DatasetWriter, Trajectory, RecoveryAudit
from .scenario import Config
from .environment import BottleneckEnv
from .expert import SequentialExpert
from .observation import policy_observation


class GapFlowScenario:
    name = 'bottleneck_family_gap1'
    recovery_protocol = 'physical gate-local perturb/re-query from successful train/dev expert anchors; test untouched'

    def __init__(self, config: Config):
        self.config = config
        self.agent_order = tuple(f'agent_{i:03d}' for i in range(config.num_agents))
        self.observation_shape = (config.num_agents, 8)
        self.action_shape = (config.num_agents, 2)
        self.environment_fingerprint = config.physical_fingerprint


def collect(config: Config, root: str | Path, *, counts: dict[str,int], seed=0,
            recovery_anchors=0):
    if set(counts) != {'train','dev','test'} or any(n < 1 for n in counts.values()):
        raise ValueError('positive train/dev/test counts required')
    root=Path(root)
    scenario=GapFlowScenario(config)
    writer=DatasetWriter(root,scenario,scenario_config=asdict(config))
    rng=np.random.default_rng(seed)
    planner=SequentialExpert()
    report={'requested':counts,'attempts':{},'successes':{},'failures':{},
            'recovery_requested':0,'recovery_success':0,'recovery_failure':{}}
    for split in ('train','dev','test'):
        done=attempts=0
        failures={}
        while done < counts[split]:
            attempts+=1
            if attempts > 5*counts[split]:
                raise RuntimeError(f'expert success too low for {split}: {failures}')
            episode_seed=int(rng.integers(0,2**31-1))
            episode_config=replace(config,seed=episode_seed,split=split)
            env=BottleneckEnv(episode_config)
            outcome=planner.rollout(env)
            if not outcome['success']:
                failures[outcome['reason']]=failures.get(outcome['reason'],0)+1
                continue
            observations=np.concatenate((outcome['observations'],
                                         policy_observation(env)[None]),axis=0)
            writer.add(Trajectory(
                rollout_id=f'{split}_gap_n{config.num_agents}_{done:04d}',
                split=split,source='nominal',
                initial_state={'positions':outcome['states'][0],
                               'goals':env.goals,'episode_seed':episode_seed,'split':split},
                states=outcome['states'],observations=observations,actions=outcome['actions'],
                metadata={'success':True,'terminal_reason':'success',
                          'expert':'sequential_grid_astar_v1',
                          'min_swept_clearance':outcome['min_clearance'],
                          'episode_seed':episode_seed,
                          'episode_steps':len(outcome['actions'])}))
            if split != 'test' and recovery_anchors:
                # Perturb real pre-terminal, pre-collision states at both gate
                # directions. Test trajectories are never recovery sources.
                states = outcome['states']; actions = outcome['actions']
                selected=[]
                for active in range(config.num_agents):
                    direction = 1 if env.goals[active,0] > config.barrier_x[0] else -1
                    entrance = config.barrier_x[0] - direction*(config.barrier_thickness/2+.06)
                    candidates = np.flatnonzero(np.linalg.norm(actions[:,active],axis=1) > .1)
                    if len(candidates):
                        chosen = int(candidates[np.argmin(abs(states[candidates,active,0]-entrance))])
                        selected.append((chosen,active))
                if len(selected) > recovery_anchors:
                    indices=np.linspace(0,len(selected)-1,recovery_anchors,dtype=int)
                    selected=[selected[i] for i in indices]
                for anchor,active in selected:
                    report['recovery_requested']+=1
                    recovery_seed=int(rng.integers(0,2**31-1))
                    local_rng=np.random.default_rng(recovery_seed)
                    recovered=None
                    for _ in range(8):
                        shifted=states[anchor].copy()
                        shifted[active]+=local_rng.normal(0,.055,2)
                        try:
                            candidate=BottleneckEnv(episode_config)
                            candidate.reset(shifted,env.goals)
                        except ValueError:
                            continue
                        candidate.velocities=(np.zeros(candidate.action_shape) if anchor==0 else
                                              actions[anchor-1].copy())
                        recovered=candidate
                        break
                    if recovered is None:
                        reason='invalid_perturbation'
                        report['recovery_failure'][reason]=report['recovery_failure'].get(reason,0)+1
                        continue
                    retried=planner.rollout(recovered)
                    if not retried['success']:
                        reason=retried['reason']
                        report['recovery_failure'][reason]=report['recovery_failure'].get(reason,0)+1
                        continue
                    observations_recovery=np.concatenate((retried['observations'],
                        policy_observation(recovered)[None]),axis=0)
                    audit=RecoveryAudit(f'{split}_gap_n{config.num_agents}_{done:04d}',split,
                                        anchor,None,None,None,recovery_seed,True)
                    writer.add(Trajectory(
                        rollout_id=f'{split}_gap_n{config.num_agents}_{done:04d}_r{active:03d}',
                        split=split,source='gate_local_recovery',
                        initial_state={'positions':retried['states'][0],
                                       'goals':recovered.goals,'episode_seed':episode_seed,
                                       'split':split,'anchor':anchor,'active_agent':active},
                        states=retried['states'],observations=observations_recovery,
                        actions=retried['actions'],
                        metadata={'success':True,'terminal_reason':'success',
                                  'expert':'sequential_grid_astar_v1',
                                  'min_swept_clearance':retried['min_clearance'],
                                  'episode_seed':episode_seed,'episode_steps':len(retried['actions'])},
                        recovery_audit=audit))
                    report['recovery_success']+=1
            done+=1
        report['attempts'][split]=attempts
        report['successes'][split]=done
        report['failures'][split]=failures
    writer.finalize(extra_report=report)
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--train',type=int,default=20)
    parser.add_argument('--dev',type=int,default=6)
    parser.add_argument('--test',type=int,default=6)
    parser.add_argument('--recovery-anchors',type=int,default=0)
    parser.add_argument('--door-width',type=float,default=.5)
    args=parser.parse_args()
    config=Config(num_agents=args.agents,max_steps=max(2000,args.agents*800),
                  openings=(((0.,args.door_width),),))
    result=collect(config,args.output,counts={'train':args.train,'dev':args.dev,'test':args.test},
                   seed=args.seed,recovery_anchors=args.recovery_anchors)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
