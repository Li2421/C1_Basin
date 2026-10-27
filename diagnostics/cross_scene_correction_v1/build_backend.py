"""Relocate frozen execution without importing historical entry points."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import difflib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    records = []

    def copy(src, dst, text=None):
        raw = src.read_bytes()
        data = raw if text is None else text.encode()
        assert dst.resolve().is_relative_to(HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert dst.read_bytes() == data, ('Existing build target differs', dst)
        else:
            dst.write_bytes(data)
        records.append(dict(source=str(src), source_sha256=sha(src), target=str(dst),
                            target_sha256=sha(dst), changed=raw != data))
        if raw != data:
            dst.with_suffix(dst.suffix + '.diff').write_text(''.join(difflib.unified_diff(
                raw.decode().splitlines(True), data.decode().splitlines(True), fromfile=str(src), tofile=str(dst))))

    # Identical environment module, with HERE naturally resolving to this experiment.
    copy(SOURCE / 'experiment.py', HERE / 'experiment.py')
    copy(SOURCE / 'bootstrap.py', HERE / 'bootstrap.py')
    pending = ['swap_runtime', 'swap_runtime_user_fix_20261023',
               'runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022']
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or name in ('bootstrap', 'experiment'):
            continue
        seen.add(name)
        src = next((p for p in (SOURCE / (name + '.py'), SOURCE / 'backend' / (name + '.py')) if p.exists()), None)
        if src is None:
            continue
        for node in ast.walk(ast.parse(src.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [v.name for v in node.names] if isinstance(node, ast.Import) else []
            pending.extend(n.split('.')[0] for n in names if n)
        # Keeping module names and all numerical code is intentional.
        copy(src, HERE / 'backend' / src.name)
    for src in (SOURCE / 'native_resume').iterdir():
        if src.suffix in ('.hpp', '.cpp'):
            old = str(SOURCE / 'runtime_cbs_query_cache')
            new = str(HERE / 'runtime_cbs_query_cache')
            copy(src, HERE / 'native_resume' / src.name, src.read_text().replace(old, new))
    for src in SOURCE.glob('*.so'):
        copy(src, HERE / src.name)
    for name in ('CIRCLE_SPEED.json', 'CONTROLLER_MANIFEST.json', 'FIXED_ETA.json',
                 'LAYOUT_REVISION_DESIGN.json', 'MOVING_USER_FIX_UNIT_GATE.json'):
        copy(SOURCE / name, HERE / name)
    (HERE / 'CIRCLE_SPEED.receipt.json').write_text(json.dumps(dict(files={str(HERE / 'CIRCLE_SPEED.json'): sha(HERE / 'CIRCLE_SPEED.json')}), indent=2) + '\n')
    command = ['g++', '-O3', '-std=c++17', '-shared', '-fPIC',
               str(HERE / 'native_resume/cbs_kernel_query_disk_candidate_20261022.cpp'),
               '-o', str(HERE / 'native_resume/cbs_kernel_resume.so'), '-lcrypto']
    subprocess.run(command, check=True)
    # The only changed native source must be the cache-root string.
    native_changes = [r for r in records if r['changed'] and '/native_resume/' in r['target']]
    assert len(native_changes) == 1 and native_changes[0]['target'].endswith('cbs_query_disk_memo_candidate_20261022.hpp')
    (HERE / 'BACKEND_BUILD.json').write_text(json.dumps(dict(sources=records, native_command=command,
        native_sha256=sha(HERE / 'native_resume/cbs_kernel_resume.so'), native_only_change='cache root'), indent=2) + '\n')
    print('Isolated backend built:', len(records), 'sources', flush=True)


if __name__ == '__main__':
    main()
