"""Seal SBGA deliverables and per-stage provenance; does not execute rollouts."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent

def main():
    def entry(p):
        return {'path': str(p.relative_to(HERE)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    stages = []
    for p in sorted((HERE / 'raw').glob('*/manifest.json')):
        m = json.loads(p.read_text())
        rows = m['records']
        stages.append({**entry(p), 'stage': m.get('stage', p.parent.name),
                       'attempts': len(rows), 'recorded_physical_steps': sum(r['steps'] for r in rows)})
    artifacts = [entry(p) for p in sorted(HERE.iterdir())
                 if p.is_file() and p.name != 'manifest.json']
    artifacts += [entry(p) for folder in ['figures', 'states'] for p in sorted((HERE / folder).glob('*')) if p.is_file()]
    manifest = {'study': 'SBGA', 'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
                'decision': json.loads((HERE / 'decision.json').read_text()),
                'verification': json.loads((HERE / 'verification.json').read_text()),
                'artifacts': artifacts, 'raw_stage_indexes': stages,
                'raw_hashes': 'Each raw manifest records individual trajectory SHA256 hashes; cache_index.json records reused provenance.',
                'frozen_configuration_and_source_provenance': 'protocol.json and verification.json; original true_q_geometry manifest hashes verified.',
                'new_recorded_attempts': sum(s['attempts'] for s in stages),
                'new_recorded_physical_steps': sum(s['recorded_physical_steps'] for s in stages),
                'unsaved_aborted_attempts_upper_bound': 48,
                'unsaved_aborted_steps_upper_bound': 24432,
                'unknown_outcomes': 'Solver failures remain null, never reassigned to the four environment outcomes.',
                'training': False, 'frozen_system_modified': False}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print({k: manifest[k] for k in ['new_recorded_attempts', 'new_recorded_physical_steps']})

if __name__ == '__main__':
    main()
