"""Versioned numerical-source manifests; release only after queued preparation."""
import guard
from guard import HERE
import datetime
import hashlib
import json
import shutil
import sys

ROOT_FILES = (
    'core.py','margin_runtime.py','physical_geometry.py','guard.py','request_factory.py','runner.py','runtime.py','admission.py',
    'routes.py','passages.py','recording_fix.py','paper_episode.py','learned_setup.py',
    'historical_replay.py','acceleration.py','scene_catalog.py',
    'PASSAGES.json','historical_reuse.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_files():
    files=[HERE/name for name in ROOT_FILES]
    for directory in ('private','generated','ports','paper_methods','accel_inputs'):
        files.extend(p for p in (HERE/directory).rglob('*') if p.is_file()
                     and p.suffix in (('.py','.so','.json') if directory=='generated' else ('.py','.so'))
                     and '__pycache__' not in p.parts)
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(set(files))}


def main():
    assert (HERE/'PRIOR_TASK_DELIVERED.json').exists()
    assert (HERE/'PORT_REPAIR_APPLIED.json').exists()
    # Materialize the deterministic generated evaluator before hashing it.
    # This imports numerical definitions but does not execute a rollout.
    import runtime
    assert __import__('pathlib').Path(runtime.__file__).resolve()==HERE/'runtime.py'
    pointer=HERE/'EXECUTION_FROZEN.json'
    files=numerical_files()
    if pointer.exists():
        previous=json.loads(pointer.read_text())
        if previous['files']==files:
            print('Existing identical numerical release', previous['revision'])
            return previous
        assert not (HERE/'PARAMETERS_FROZEN.json').exists(),'No numerical release changes after parameter freeze'
        assert not (HERE/'FORMAL_TEST_STARTED.json').exists(),'No silent post-test port repair'
        revision=previous['revision']+1
    else:
        revision=1
    folder=HERE/'execution_revisions'/f'v{revision:03d}'
    folder.mkdir(parents=True,exist_ok=False)
    for relative,expected in files.items():
        target=folder/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE/relative,target)
        assert sha(target)==expected
    value=dict(released_at=datetime.datetime.now().astimezone().isoformat(),revision=revision,
               files=files,immutable_manifest=str((folder/'MANIFEST.json').relative_to(HERE)),
               archived_source=str((folder/'source').relative_to(HERE)),
               input_snapshot_sha256=sha(HERE/'INPUT_SNAPSHOT.json'),
               note='Old source snapshots and requests retained. Changed pre-test sources create new UIDs; repaired-paper results never inherit old repaired-port labels.')
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    (folder/'MANIFEST.json').write_text(text)
    temporary=pointer.with_suffix('.tmp')
    temporary.write_text(text)
    temporary.replace(pointer)
    if 'request_factory' in sys.modules:
        sys.modules['request_factory'].sources.cache_clear()
    print('Numerical release',revision,'files',len(files),'manifest',sha(pointer))
    return value


if __name__=='__main__':main()
