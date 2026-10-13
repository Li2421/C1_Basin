"""Static sufficiency falsification; no physical simulation or training.

For the existing endpoint-matched affine family, single-key field probes
cannot identify the Gaussian initializer law. This tests the precise alias,
not whether it explains existing LOSO errors or changes terminal Q.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from diagnostics.controller_q_diversity_v1.experiment import read, write, freeze, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'controller_four_scene_mechanism_v1'
FAMILIES = ('isotropic', 'anisotropic', 'rotate_positive', 'rotate_negative')
SCENES = ('toy_give_way', 'double_bottleneck', 'four_way_intersection', 'ring_exchange')


def prepare():
    from shared_rollout_db.src.rollout_db import connect
    parent = read(BASE / 'protocol.json')
    states = []
    for scene in SCENES:
        ss = [s for s in parent['states'] if s['scenario'] == scene and s['study_split'] == 'train']
        states.extend(sorted(ss, key=lambda s: s['study_index'])[:8])
    assert len(states) == 32
    inventory = []
    with connect(True) as db:
        for r in db.execute('SELECT controller_uid,config_json FROM controller_config'):
            p = json.loads(r['config_json'])
            inventory.append(dict(controller_uid=r['controller_uid'], config=p))
    p = dict(schema='exact_single_key_initializer_alias_static_v1', states=states,
        families=FAMILIES, scenes=SCENES, alpha=2., descriptor_key=2026100908,
        independent_action_key=2026101204, independent_action_folds=[0,1,2,3],
        eta_indices=[0,5,10,15], eta=parent['eta'],
        construction='scale_prime=2*scale; mean_prime=mean+(scale-scale_prime)*z_descriptor. Same reference and M; only wrapper initializer changes. Base Flow unmodified.',
        identity='At descriptor key, h0 is identical in every physical pose. Endpoint reference, M, center, every physical field probe and nominal14 are consequently identical in exact arithmetic. At other keys h0 generally differs.',
        selection='First8 natural original TRAIN states per scene in original generator order; ALL4 uncoupled M families. No success/critic/candidate filtering.',
        reuse='All original nominal/field fingerprints and static independent-key FF actions reused. One original action/descriptor replay per scene/family verifies compatibility. No old rollout rerun.',
        old_tests=[dict(path='controller_program_input_probe_v1', distinction='Already tested exact6+initializer8 and field+initializer8; negative transfer. Not repeated; no claim initializer augmentation alone repairs LOSO.'),
                   dict(path='controller_operator_sufficiency_localization_v1', distinction='Current M recoverability already verified; this tests stochastic initialization information, not missing M.'),
                   dict(path='controller_ff_action_mediation_v1', distinction='Original independent-key action cache reused; no modified-law alias pairs found in historical evidence index or full controller inventory.')],
        boundary='Analytic adversarial sufficiency test over a broader controller class, not an in-distribution collision or proof of current LOSO root cause. No terminal-Q claim. Existing scene-confounded initializer distributions require a separate causal Q contrast if justified.',
        literature=[dict(url='https://arxiv.org/pdf/2002.11833', relevance='Policy fingerprints query policy outputs at probing states. Does not guarantee our single-random-key FF field identifies a stochastic program.')],
        numerical_tolerance=1e-5, simulator_steps=0, training_fits=0,
        source_protocol_sha256=sha(BASE/'protocol.json'), code_sha256=sha(__file__))
    freeze(HERE/'protocol.json', p)
    freeze(HERE/'cache_history_audit.json', dict(profiles_scanned=len(inventory), inventory=inventory,
        history_ledger_sha256=sha(ROOT/'controller_cross_scene_root_v2/TEST_AND_CAUSE_LEDGER.json'),
        new_terminal_labels=0, database_writes=0, exact_execution_old_actions_reused=2048))


def worker(index):
    import jax
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime, controller_context
    from diagnostics.controller_response_families_v1.runtime import EndpointMatchedField
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    p = read(HERE/'protocol.json')
    assert sha(__file__) == p['code_sha256']
    scene = SCENES[index]
    states = [s for s in p['states'] if s['scenario'] == scene]
    base = native_runtime(scene)
    mean, scale = base.mean.copy(), base.scale.copy()
    z = np.asarray(jax.random.normal(jax.random.PRNGKey(p['descriptor_key']),
                   (1, base.agent.config['joint_act_dim']))).reshape(-1,2)
    rows = []
    for si, s in enumerate(states):
        for family in FAMILIES:
            # Descriptor cache uses native precision; independent action cache
            # explicitly used x64=False even for Toy. Keep both identities.
            jax.config.update('jax_enable_x64', scene == 'toy_give_way')
            old = EndpointMatchedField(base, family)
            alt = EndpointMatchedField(base, family)
            alt.scale = p['alpha'] * scale
            alt.mean = mean + (scale-alt.scale)*z
            # Assign new arrays: never mutate the frozen base's mean/scale.
            np.testing.assert_array_equal(base.mean, mean)
            np.testing.assert_array_equal(base.scale, scale)
            env = old.make_env(s)
            def forbidden(*args, **kwargs):
                raise AssertionError('Physical simulation prohibited in static alias audit')
            env.step = forbidden
            np.testing.assert_allclose(old.to_world(mean+scale*z,env),
                alt.to_world(alt.mean+alt.scale*z,env), atol=1e-12, rtol=0)
            cached = dict(np.load(BASE/'features'/s['state_uid']/(family+'.npz')))
            class Interface:
                def __init__(self, rt): self.rt=rt
                def __getattr__(self, name): return getattr(self.rt,name)
                def make_env(self, state):
                    e=self.rt.make_env(state); e.step=forbidden; return e
            nominal = controller_context(Interface(alt), s)
            field = field_context(Interface(alt), s)
            original_raw = np.concatenate((cached['nominal'],cached['field']),-1)
            alternate_raw = np.concatenate((nominal,field),-1)
            a = np.asarray(compact_field(original_raw))
            b = np.asarray(compact_field(alternate_raw))
            if si == 0:
                np.testing.assert_allclose(controller_context(Interface(old),s),cached['nominal'],atol=1e-6,rtol=0)
                np.testing.assert_allclose(field_context(Interface(old),s),cached['field'],atol=1e-6,rtol=0)
            ref = dict(np.load(ROOT/'controller_ff_action_mediation_v1/features'/s['state_uid']/(family+'.npz')))
            changed = np.full_like(ref['raw_actions'],np.nan)
            valid = np.zeros_like(ref['query_success'],bool)
            errors = []
            jax.config.update('jax_enable_x64', False)
            for j in p['eta_indices']:
                for k in p['independent_action_folds']:
                    key=jax.random.fold_in(jax.random.PRNGKey(p['independent_action_key']),k)
                    if si == 0 and j == 0 and k == 0:
                        u=old.action(env,key,np.asarray(p['eta'][j]),'field')[0]
                        np.testing.assert_allclose(u,ref['raw_actions'][j,k],atol=2e-6,rtol=0)
                    try:
                        changed[j,k]=alt.action(env,key,np.asarray(p['eta'][j]),'field')[0]
                        valid[j,k]=True
                    except Exception as ex:
                        errors.append(dict(eta_index=j,seed=k,type=type(ex).__name__,message=str(ex)[:400]))
            dest=HERE/'raw'/s['state_uid']/(family+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,original_raw=original_raw,alternate_raw=alternate_raw,
                original_compact=a,alternate_compact=b,original_actions=ref['raw_actions'],
                alternate_actions=changed,original_valid=ref['query_success'],alternate_valid=valid,
                original_mean=mean,original_scale=scale,alternate_mean=alt.mean,alternate_scale=alt.scale)
            ix=np.asarray(p['eta_indices']);both=ref['query_success'][ix]&valid[ix]
            delta=np.max(np.abs(changed[ix]-ref['raw_actions'][ix]),axis=(-2,-1))/base.config.max_speed
            rows.append(dict(scene=scene,state_uid=s['state_uid'],controller=family,
                compact_max_error=float(np.max(np.abs(a-b))),
                raw_max_error=float(np.max(np.abs(original_raw-alternate_raw))),
                action_cases=int(both.sum()),changed_action_cases=int(np.sum(delta[both]>1e-5)),
                action_max_difference_vmax=float(delta[both].max()) if both.any() else None,
                errors=errors))
    write(HERE/f'results{index}.json',dict(rows=rows,scene=scene,physical_steps=0,new_terminal_labels=0))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker']);ap.add_argument('--index',type=int,default=0)
    args=ap.parse_args()
    prepare() if args.action=='prepare' else worker(args.index)
