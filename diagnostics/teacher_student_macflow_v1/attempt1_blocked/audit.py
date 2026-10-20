"""Read-only static protocol audit; writes only alongside this script."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT.parent / '01_MACFlow_Baseline_Reproduction/MACFlow_Official'
FIELD = ROOT.parent / 'Basin_C1_flow_field_poc_20261004'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def snapshot():
    result = {}
    for name, path in [('Basin_C1', ROOT), ('MACFlow_Official', OFFICIAL)]:
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=path)
        result[name] = dict(path=str(path), head=git('rev-parse', 'HEAD').decode().strip(),
                            status_porcelain=git('status', '--porcelain').decode(),
                            diff_sha256=hashlib.sha256(git('diff')).hexdigest())
    result['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result

sources = [ROOT / p for p in [
    'AGENTS.md', 'shared_rollout_db/README.md',
    'docs/prompt_round30_teacher_student_macflow_20261015.md',
    'docs/controller_generalization_ring_summary_20261014.md',
    'diagnostics/decentralized_pipeline_v1/core.py',
    'diagnostics/decentralized_pipeline_v1/pipeline.py',
    'diagnostics/decentralized_pipeline_v1/run.py',
    'diagnostics/decentralized_pipeline_v1/REPORT.md',
    'diagnostics/decentralized_pipeline_v1/main_protocol.json',
    'diagnostics/decentralized_pipeline_v1/IMPLEMENTATION_FROZEN.receipt.json',
    'diagnostics/other_controller_classes_v1/probes.py',
    'diagnostics/other_controller_classes_v1/freeze_choices.py',
    'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py',
    'diagnostics/controller_q_diversity_v1/runtime.py',
    'ring_exchange/local_frame.py', 'new_benchmark_common/macflow.py',
    'toy_giveway/flowbc.py', 'double_bottleneck/flowbc_4a_agent.py']]
sources += [OFFICIAL / 'agents/macflow.py', FIELD / 'poc/field_controller.py',
            FIELD / 'validation_stage2/run_rollouts.py',
            ROOT.parent / '02_C1_Toy_GiveWay/flowbc/giveway_flowbc_agent.py']
write('SOURCE_HASHES.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    audit_method='static reads and AST only; no existing project source imported or copied',
    imported_existing_sources=[], copied_existing_sources=[],
    read_existing_sources={str(p): sha(p) for p in sources}))
protocol = json.loads((ROOT / 'diagnostics/decentralized_pipeline_v1/main_protocol.json').read_text())
checkpoints = []
for name, profile in protocol['base_profiles'].items():
    payload = profile['payload']; path = Path(payload['flow_checkpoint_path'])
    actual = sha(path)
    checkpoints.append(dict(name=name, path=str(path), expected_sha256=payload['flow_checkpoint_sha256'],
                            actual_sha256=actual, passed=actual == payload['flow_checkpoint_sha256']))
write('TEACHER_CHECKPOINT_AUDIT.json', dict(count=len(checkpoints), all_passed=all(c['passed'] for c in checkpoints), checkpoints=checkpoints))
receipt = json.loads((ROOT / 'diagnostics/decentralized_pipeline_v1/IMPLEMENTATION_FROZEN.receipt.json').read_text())
write('ROUND28_FROZEN_AUDIT.json', {p: dict(expected_sha256=h, actual_sha256=sha(p), passed=sha(p)==h) for p,h in receipt['files'].items()})
interfaces=[]
for path, wanted in [(FIELD/'poc/field_controller.py', {'nominal'}),
                     (ROOT/'new_benchmark_common/macflow.py', {'sample_actions','sample_bounded_actions'}),
                     (ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py', {'trace','velocity'}),
                     (OFFICIAL/'agents/macflow.py', {'compute_flow_actions'})]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            interfaces.append(dict(path=str(path), function=node.name, line=node.lineno,
                                   signature=ast.unparse(node.args), source=ast.get_source_segment(path.read_text(),node)))
write('INTERFACE_AUDIT.json', dict(status='BLOCKED_REQUIRED_INTERFACE_MISSING',
    required='round28 runtime sampling path accepting explicit joint xi', interfaces=interfaces,
    call_chain=['Runtime.nominal(env,key,traced=False)', 'native_fn(obs,key)',
                'sample_bounded_actions(agent,observations,seed)', 'JointMACFlowAgent.sample_actions(observations,seed)',
                'jax.random.normal(seed,(batch,joint_act_dim))'],
    note='Official MACFlowAgent.compute_flow_actions accepts noises but is not the frozen Ring teacher runtime. velocity_fn accepts z for a single vector-field query, not the runtime sampler.'))
end = snapshot(); write('INTEGRITY_END.json', end)
start = json.loads((HERE/'INTEGRITY_START.json').read_text())
def filtered(status):
    return '\n'.join(line for line in status.splitlines() if 'diagnostics/teacher_student_macflow_v1' not in line)
comparisons={}
for name in ['Basin_C1','MACFlow_Official']:
    comparisons[name] = dict(head_unchanged=start[name]['head']==end[name]['head'],
       diff_sha256_unchanged=start[name]['diff_sha256']==end[name]['diff_sha256'],
       status_outside_new_directory_unchanged=filtered(start[name]['status_porcelain'])==filtered(end[name]['status_porcelain']),
       exact_status_unchanged=start[name]['status_porcelain']==end[name]['status_porcelain'])
passed=all(v['head_unchanged'] and v['diff_sha256_unchanged'] and v['status_outside_new_directory_unchanged'] for v in comparisons.values())
write('INTEGRITY_COMPARISON.json',dict(passed=passed,allowed_status_difference='new diagnostics/teacher_student_macflow_v1/ only',repositories=comparisons))
print(json.dumps(dict(integrity_passed=passed,teachers=len(checkpoints),teacher_hashes_passed=all(c['passed'] for c in checkpoints)),indent=2))
