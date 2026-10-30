"""Pure request/RNG regressions against copied original factories; no rollout."""
import guard
from guard import HERE,ROOT
import ast
import hashlib
import json
from pathlib import Path
import sys
from functools import lru_cache

sys.path.insert(0,str(ROOT))
from shared_rollout_db.src import rollout_db as db

BASE=HERE/'inputs/diagnostics'
HETERO=BASE/'hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def main():
    common=dict(json=json,db=db,Path=Path,HERE=HERE,read=read,sha=sha,lru_cache=lru_cache,EXP='unit_request_identity_no_registration',
                sources=lambda:dict(unit_test_only=True,freeze_sha256='unit_only'))
    fresh=dict(common)
    load_functions(HERE/'request_factory.py',{'legacy_a_rng','certified_acceleration','case'},fresh)
    old_a=dict(common,dbm=db,uid=db.uid,canonical=db.canonical,
               HERE=HETERO/'scene_search/harness')
    load_functions(HETERO/'scene_search/harness/common.py',{'make_case'},old_a)
    load_functions(HETERO/'scene_search/harness/scenes.py',{'make_new'},old_a)
    old_b=dict(common,EXP='unit_request_identity_no_registration')
    load_functions(BASE/'scene_types_oracle_coverage_v3/runner.py',{'case'},old_b)
    scenes_a=read(HETERO/'hetset/scenes_all2.json')
    scenes_b=[e['scene']for e in read(BASE/'scene_types_oracle_coverage_v3/SCENES.json')['entries']]
    controller=dict(type_code='unit_frozen_nominal',checkpoint='unit_only',parameters='[]')
    spec=dict(eta=[.875,0.,.375],coefficients={})
    action=dict(key='unit_eta',eta_a=spec['eta'],eta_b=spec['eta'],theta=None)
    checks=[]
    for benchmark,scenes in (('A',scenes_a),('B',scenes_b)):
        for scene in scenes:
            new=fresh['case'](benchmark,scene,controller,0,spec=spec,accelerated=False,
                              phase='training',purpose='identity_unit')
            if benchmark=='A':
                old=old_a['make_new'](scene,controller,scene['states'][0],action,0)
                expected_rng=old['state_uid']
            else:
                old=old_b['case'](scene,controller,0,spec=spec)
                expected_rng=old['rng_state_uid']
            assert new['rng_state_uid']==expected_rng,(benchmark,scene['name'],'RNG changed')
            other=fresh['case'](benchmark,scene,controller,0,spec=spec,accelerated=False,
                                phase='certification',purpose='other_analysis_label')
            assert new['rollout_uid']==other['rollout_uid'],'Analysis purpose/phase entered numerical identity'
            if not(HERE/'ACCELERATION_CERTIFICATE.json').exists():
                default=fresh['case'](benchmark,scene,controller,0,spec=spec)
                assert default['rollout_uid']==new['rollout_uid'],'Uncertified acceleration enabled by default'
            altered=dict(eta=[.5,0.,.375],coefficients={})
            different=fresh['case'](benchmark,scene,controller,0,spec=altered,accelerated=False)
            assert different['rollout_uid']!=new['rollout_uid'],'Correction coefficient absent from identity'
            if (HERE/'PARAMETERS_FROZEN.json').exists():
                test_freeze_guard='Not tested after real parameter freeze'
            else:
                try:
                    fresh['case'](benchmark,scene,controller,0,spec=spec,accelerated=False,phase='test')
                except AssertionError:test_freeze_guard=True
                else:raise AssertionError('Formal test allowed before parameter/analysis freeze')
            checks.append(dict(benchmark=benchmark,scene=scene['name'],rng_preserved=True,
                               analysis_labels_not_execution_identity=True,eta_changes_identity=True,
                               formal_test_guard=test_freeze_guard))
    output=dict(passed=True,scenes=len(checks),records=checks,new_rollouts=0,
        original_source_hashes={str(path.relative_to(HERE)):sha(path)for path in
            (HERE/'request_factory.py',HETERO/'scene_search/harness/common.py',HETERO/'scene_search/harness/scenes.py',BASE/'scene_types_oracle_coverage_v3/runner.py')},
        scope='Actual request factory functions with official pure UID helpers, a test-only source manifest and dummy nominal row. No model/controller execution, no registration, no execution equivalence claim.')
    (HERE/'REQUEST_IDENTITY_UNIT_CHECKS.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Request identity/RNG/phase regressions passed for',len(checks),'copied scenes; zero rollouts.')


if __name__=='__main__':main()
