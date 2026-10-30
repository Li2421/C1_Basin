"""Prepare private copies and path adapters, without importing simulation code."""
import guard
from guard import HERE, ROOT
import ast
import datetime
import hashlib
import json
from pathlib import Path

BASE = HERE / 'inputs/diagnostics/scene_types_oracle_coverage_v3'
PRIVATE = HERE / 'private/Basin_C1'
RECORDS = []


def write_adapted(target, data):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_text() == data, ('Existing adapter must not be overwritten', target)
    else:
        target.write_text(data)


def copy(source, target):
    source, target = Path(source), Path(target)
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == data, ('Immutable execution input changed', source, target)
    else:
        target.write_bytes(data)
    RECORDS.append(dict(source=str(source), target=str(target), bytes=len(data),
                        source_sha256=hashlib.sha256(data).hexdigest(), target_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


def main():
    # Load the official DB before adding this private namespace to sys.path in
    # core.py. Everything else below is a copied research dependency.
    for package in ('shared_control', 'new_benchmark_common'):
        for source in sorted((ROOT / package).rglob('*.py')):
            if '__pycache__' not in source.parts:
                copy(source, PRIVATE / source.relative_to(ROOT))
    for package in ('bottleneck_family', 'single_integrator'):
        source_root = HERE / 'inputs/production'
        for source in sorted((source_root / package).rglob('*.py')):
            copy(source, PRIVATE / source.relative_to(source_root))
    source = ROOT / 'diagnostics/double_bottleneck_eta3_basin/tools/exact_projection_retry.py'
    copy(source, PRIVATE / source.relative_to(ROOT))
    for folder in ('diagnostics', 'diagnostics/double_bottleneck_eta3_basin', 'diagnostics/double_bottleneck_eta3_basin/tools'):
        path = PRIVATE / folder / '__init__.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('"""Private dependency namespace; no external diagnostics imports."""\n')
    utility_source = ROOT.parent / '01_MACFlow_Baseline_Reproduction/MACFlow_Official/utils'
    utility_dest = HERE / 'private/01_MACFlow_Baseline_Reproduction/MACFlow_Official/utils'
    for source in sorted(utility_source.glob('*.py')):
        copy(source, utility_dest / source.name)
    # This layout leaves macflow.py byte-identical: its original sibling-project
    # lookup now resolves to copied utility files.
    for source in sorted((BASE / 'generated').rglob('*')):
        if source.is_file() and source.suffix in ('.py', '.json', '.so', '.cpp', '.hpp', '.h'):
            copy(source, HERE / 'generated' / source.relative_to(BASE / 'generated'))
    copy(BASE / 'accel_inputs/fastproj.py', HERE / 'accel_inputs/fastproj.py')
    copy(BASE / 'accel_inputs/accel.py', HERE / 'accel_inputs/accel.py')
    copy(BASE / 'accel_inputs/np_threefry.py', HERE / 'accel_inputs/np_threefry.py')
    for source in sorted((BASE / 'ports').glob('*.py')):
        # These two modules contain historical absolute data roots. Change only
        # that literal; author/vendor implementation files remain unmodified.
        target = HERE / 'ports' / source.name
        data = source.read_text()
        anchor = "HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/paper_baselines_20261028')"
        edits = 0
        if source.name in ('paper_adapters.py', 'impc_adapter.py'):
            assert data.count(anchor) == 1
            data = data.replace(anchor, "HERE=Path(__file__).resolve().parents[1]/'inputs/diagnostics/paper_baselines_20261028'")
            edits = 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_text() == data, ('Path adapter differs', target)
        else:
            target.write_text(data)
        RECORDS.append(dict(source=str(source), target=str(target), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), path_literal_edits=edits,
                            numerical_statements_changed=False))
    admission_base = HERE / 'inputs/diagnostics/bottleneck_one_way_admission_v1'
    for name in ('routes.py', 'admission.py', 'PASSAGES.json'):
        copy(admission_base / name, HERE / name)
    source = admission_base / 'runtime.py'
    data = source.read_text()
    anchor = "target=HERE/'generated/route_dictionary_evaluate.py'"
    assert data.count(anchor) == 1
    data = data.replace(anchor, "target=HERE/'generated/admission_route_dictionary_evaluate.py'")
    target = HERE / 'runtime.py'
    write_adapted(target, data)
    RECORDS.append(dict(source=str(source), target=str(target), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        edit='Generated diagnostic-module output path only; original control arithmetic literal.'))
    copy(BASE / 'recording_fix.py', HERE / 'recording_fix.py')
    source = BASE / 'paper_episode.py'
    tree = ast.parse(source.read_text())
    original_run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    # Keep the entire numerical run() literal. Only the entrypoint and private
    # import path are changed for the new scheduler/runner.
    data = '''"""Private copy of the v3 common-plant paper episode loop."""\nimport sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom core import *\nimport os,time,traceback\nfrom routes import guidance,route_geometry\nimport importlib.util\n\n'''
    data += ast.get_source_segment(source.read_text(), original_run) + '\n\n'
    data += '''if __name__ == '__main__':
    import runner
    from budget import reserve
    os.environ.pop('STB_COORDINATOR',None)
    request=read(sys.argv[1])
    reserve(request)
    result=runner.work(request)
    print('paper',result['controller'],result['scene'],result['state_index'],result['outcome'],round(result['wall_seconds'],3),flush=True)
'''
    target = HERE / 'paper_episode.py'
    write_adapted(target, data)
    copied_run = next(node for node in ast.parse(data).body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    assert ast.dump(original_run) == ast.dump(copied_run)
    RECORDS.append(dict(source=str(source), target=str(target), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        edit='Private import path and coordinator entrypoint only.', numerical_run_ast_identical=True))
    output = dict(prepared_at=datetime.datetime.now().astimezone().isoformat(), sources=RECORDS,
                  source_root=str(PRIVATE), official_database_root=str(ROOT / 'shared_rollout_db'),
                  new_rollouts=0, certified_for_rollout=False,
                  next_step='Pin official database, preload copied packages under canonical names, then perform mandatory execution equivalence checks before reuse/acceleration.')
    (HERE / 'EXECUTION_COPY.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('Private execution files prepared:', len(RECORDS), '; no imports of copied research code; no rollout')


if __name__ == '__main__':
    main()
