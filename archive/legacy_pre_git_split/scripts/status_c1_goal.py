"""Compact read-only status for the ongoing C1 research runs."""
import json
from pathlib import Path


def main():
    roots = ['c1_certificate_development', 'c1_completion_development', 'c1_independent_distribution', 'c1_deadlock_primary', 'c1_deadlock_union', 'c1_scene_deadlock_union']
    for root in roots:
        for path in sorted((Path('results')/root).glob('**/config.json')):
            folder = path.parent
            config = json.loads(path.read_text())
            history = json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
            validation = json.loads((folder/'validation.json').read_text()) if (folder/'validation.json').exists() else []
            calibration = json.loads((folder/'calibration.json').read_text()) if (folder/'calibration.json').exists() else {}
            print(json.dumps(dict(run=str(folder), objective=config.get('objective'),
                updates=len(history), target=config.get('updates'), complete=(folder/'complete.json').exists(),
                feasible=(folder/'best_feasible.pkl').exists(),
                calibration={k:calibration[k] for k in ('J_live','deadlock','stalled_deadlock','either_deadlock','timeout','success') if k in calibration},
                accepted=sum(r['step_control']['status']=='accepted' for r in history),
                restarted=sum(r['step_control'].get('proposal')=='restart' for r in history),
                validation=[{k:r[k] for k in ['update','J_live','J_def','success','deadlock','stalled_deadlock','either_deadlock','timeout'] if k in r} for r in validation])))


if __name__ == '__main__':
    main()
