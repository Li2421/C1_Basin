"""Prepare isolated historical runtime; no controller imports or simulation."""
from pathlib import Path
import datetime
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'diagnostics/blackbox_families_v1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    assert not (HERE / 'COPIED_SOURCES.json').exists()
    copies = {}
    for name in ('core.py', 'common.py', 'legacy_controllers.py', 'historical_controllers.py',
                 'gaussian.py', 'controllers.py', 'learned.py', 'episode.py', 'right_hand.py',
                 'trigger.py', 'progress.py'):
        source = OLD / name
        original = source.read_text()
        content = original
        if name == 'common.py':
            content = content.replace("EXP='exp_blackbox_families_v1'", "EXP='exp_crosstype1_learned_zoo_selector_v1'")
        target = HERE / name
        assert not target.exists()
        target.write_text(content)
        copies[name] = dict(source=str(source), source_sha256=sha(source), copy_sha256=sha(target),
                            diff=list(difflib.unified_diff(original.splitlines(), content.splitlines(),
                                                          fromfile=str(source), tofile=str(target))))
    write('COPIED_SOURCES.json', copies)
    # The 24h audit is deliberately conservative. This finished, explicitly
    # requested historical source has no matching running Slurm task.
    sources = {str(OLD / k) for k in copies}
    imported = json.loads((OLD / 'IMPORTED_SOURCES.json').read_text())
    sources.update(row['path'] for row in imported['sources'])
    for folder in (ROOT / 'shared_rollout_db', ROOT / 'new_benchmark_common', ROOT / 'ring_exchange',
                   ROOT / 'shared_control'):
        sources.update(str(p) for p in folder.rglob('*.py') if '__pycache__' not in p.parts)
    write('PREIMPORT_SOURCES.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          sources={p:sha(p) for p in sorted(sources)},
          historical_source_exception='Finished round38 explicitly required by prompt; 24h metadata candidate, not an active worker tree.',
          imports_started=False))
    inventory = {}
    for name in ('DESIGN_CONFIG.json', 'PILOT_STATES.json', 'TUNING_STATES.json', 'EVAL_STATES.json',
                 'LEARNED_UNIT_TESTS.json', 'FRAME_TESTS.json', 'EXPANDED_STEP_SELECTION.json'):
        source = OLD / name
        value = json.loads(source.read_text())
        inventory[str(source)] = dict(sha256=sha(source), count=len(value), bytes=source.stat().st_size)
    for name in ('PILOT_STATES.json', 'TUNING_STATES.json', 'EVAL_STATES.json'):
        (HERE / name).write_bytes((OLD / name).read_bytes())
    probe = ROOT / 'diagnostics/controller_flow_blend_v1/stage_2/probes/protocol.json'
    values = json.loads(probe.read_text())
    assert len(values['states']) == 16
    write('PROBE_STATES.json', values['states'])
    states = {k:json.loads((HERE / k).read_text()) for k in
              ('PILOT_STATES.json', 'TUNING_STATES.json', 'EVAL_STATES.json', 'PROBE_STATES.json')}
    assert [len(states[k]) for k in states] == [16,16,32,16]
    def physical(s):
        return json.dumps({k:s['physical'][k] for k in ('positions','velocities','goals')}, sort_keys=True)
    seen = {}
    overlaps = []
    for split, rows in states.items():
        for row in rows:
            key = physical(row)
            if key in seen:
                overlaps.append([seen[key], split, row['uid']])
            seen[key] = split
    assert not overlaps, overlaps
    oldt = json.loads((ROOT / 'diagnostics/blackbox_addendum_v1/TUNING_STATES.json').read_text())
    assert [physical(s) for s in oldt] == [physical(s) for s in states['TUNING_STATES.json']]
    write('STATE_SPLIT_AUDIT.json', dict(counts={k:len(v) for k,v in states.items()},
          overlaps=overlaps, tuning_matches_round37=True, files={k:sha(HERE/k) for k in states},
          probe_source=str(probe), probe_source_sha256=sha(probe), new_states_generated=0))
    # Read the metadata of all required frozen arrays; no result calculation.
    import numpy as np
    for name in ('EVALUATION_LABELS.npz', 'EXPANDED_TRAINING_DATA.npz'):
        p = OLD / name
        with np.load(p, allow_pickle=False) as data:
            inventory[str(p)] = dict(sha256=sha(p), arrays={k:dict(shape=list(data[k].shape),dtype=str(data[k].dtype)) for k in data.files})
    for folder in (OLD / 'policies', OLD / 'expanded'):
        inventory[str(folder)] = {str(p):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    write('MANDATORY_ARTIFACT_READING.json', inventory)
    print(json.dumps(dict(copies=len(copies), preimport_sources=len(sources), state_split='passed')))

if __name__ == '__main__':
    main()
