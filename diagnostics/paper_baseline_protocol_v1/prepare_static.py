"""Copy historical evidence and execution inputs; never launch a rollout."""
import guard
from pathlib import Path
import datetime, hashlib, json, shutil, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARALLEL = [Path(x['path']).resolve() for x in json.loads((HERE/'PARALLEL_TASKS.json').read_text())['parallel_directories']]
MANIFEST = HERE/'INPUT_SNAPSHOT.json'
records = json.loads(MANIFEST.read_text())['files'] if MANIFEST.exists() else {}

def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''): h.update(data)
    return h.hexdigest()

def copy(relative, target=None):
    source = (ROOT/relative).resolve()
    assert not any(source.is_relative_to(p) for p in PARALLEL), source
    assert source.is_file(), source
    dest = HERE/'inputs'/(target or relative)
    assert dest.resolve().is_relative_to(HERE)
    before = checksum(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert checksum(dest) == before, ('Immutable input changed', source, dest)
    else:
        shutil.copyfile(source, dest)
    assert checksum(source) == before == checksum(dest)
    records[str(dest.relative_to(HERE))] = dict(source=str(source), source_sha256=before,
        target=str(dest), target_sha256=before, bytes=dest.stat().st_size)

def main():
    sources = {
        'diagnostics/paper_baselines_20261028': ['report.md','PAPER_REGISTRY.json','SOURCE_PINS.json','paper_design.py'],
        'diagnostics/scene_types_oracle_coverage_v3': ['PAPER_FAIRNESS.json','PAPER_PORT_INPUTS.json','SCENES.json','REPRESENTATIVE17_FROZEN.json','FAMILIES_64.json','ACCELERATION_CERTIFICATE.json','SCENE2_RESOLUTION.json','generated/LAYOUT_REVISION_DESIGN.json'],
        'diagnostics/scene_types_oracle_coverage_v3/paper_after_four_column': ['REPORT.md','RESULTS.json','PARAMETER_PROVENANCE.md','TIME_COMPARISON_DIAGNOSIS.json','PROFILES_FROZEN.json','COMPLETED_COMPARISON.json'],
        'diagnostics/hetero_set_critic_selfcheck_v1': ['SPLIT.json','FINAL_FROZEN.json'],
        'diagnostics/hetero_set_learned_planners_v1': ['README.md','LEARNED.json','FROZEN_T2.json','eval.jsonl','t2_eval.jsonl'],
        'diagnostics/bottleneck_one_way_admission_v1': ['FAMILIES_THIS_ROUND.json','PASSAGES.json','DESIGN_FREEZE_v2.json','runtime.py','admission.py','passages.py','core.py','routes.py','runner.py'],
        'diagnostics/scene_types_basis_dimension_v1': ['scenes.py','prepare.py','core.py'],
        'docs': ['prompt_scene_types_v2_paper_baselines_20261028.md','exact_engineering_acceleration_20261026.md','video_delivery_optional.md'],
        'shared_rollout_db': ['README.md'],
    }
    for folder, names in sources.items():
        for name in names: copy(folder+'/'+name)
    base = ROOT/'diagnostics/scene_types_oracle_coverage_v3'
    for path in sorted(base.glob('*.py')): copy(str(path.relative_to(ROOT)))
    for folder in ['generated','ports','paper_methods','accel_inputs']:
        for path in sorted((base/folder).rglob('*.py')): copy(str(path.relative_to(ROOT)))
    for path in sorted((base/'forest_layouts').glob('*.json')): copy(str(path.relative_to(ROOT)))
    design = ROOT/'diagnostics/scene_types_basis_dimension_v1/scene_design'
    for path in sorted(design.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.md'): copy(str(path.relative_to(ROOT)))
    for package in ['bottleneck_family','single_integrator']:
        for path in sorted((ROOT/package).rglob('*.py')):
            if '__pycache__' not in path.parts:
                copy(str(path.relative_to(ROOT)), 'production/'+str(path.relative_to(ROOT)))
    if '--large' in sys.argv:
        folders = [
            ROOT/'diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027',
            ROOT/'diagnostics/hetero_set_learned_planners_v1/code',
            ROOT/'diagnostics/hetero_set_learned_planners_v1/checkpoints',
            ROOT/'diagnostics/paper_baselines_20261028/vendor',
        ] + sorted((ROOT/'diagnostics/paper_baselines_20261028').glob('deps*'))
        for folder in folders:
            if not folder.is_dir(): continue
            for path in sorted(folder.rglob('*')):
                if path.is_file() and not any(x in path.parts for x in ('__pycache__','.git','.hg')):
                    copy(str(path.relative_to(ROOT)))
            print('Snapshot copied:',folder.relative_to(ROOT),flush=True)
    data = dict(time=datetime.datetime.now().astimezone().isoformat(), files=records,
        stage='static preparation; no rollout or parameter selection',
        parallel_directories=[str(p) for p in PARALLEL],
        relocated_references={
            'v3/PARAMETER_PROVENANCE.md':'v3/paper_after_four_column/PARAMETER_PROVENANCE.md',
            'v3/TIME_COMPARISON_DIAGNOSIS.json':'v3/paper_after_four_column/TIME_COMPARISON_DIAGNOSIS.json'},
        new_rollouts=0)
    MANIFEST.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Copied',len(records),'immutable input files;',sum(r['bytes']for r in records.values()),'bytes; rollouts0')

if __name__ == '__main__': main()
