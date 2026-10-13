"""Read-only inventory of every research rollout database before acquisition."""
from pathlib import Path
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2]
MAIN = HERE.parents[1]


def main():
    paths = sorted({p for pattern in ('*.sqlite', '*.db')
                    for p in RESEARCH.rglob(pattern)
                    if '/.git/' not in str(p) and '/.venv' not in str(p)})
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'new_rollouts': 0, 'databases': [],
              'physical_encoding': str(MAIN/'diagnostics/physical_state_critic_formal_v1/physical_entities.py')}
    result['physical_encoding_sha256'] = hashlib.sha256(Path(result['physical_encoding']).read_bytes()).hexdigest()
    for p in paths:
        con = sqlite3.connect('file:'+str(p)+'?mode=ro', uri=True)
        con.execute('BEGIN')
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        row = {'path': str(p), 'bytes': p.stat().st_size, 'tables': tables}
        if 'controller_config' in tables:
            counts = {r[0]: dict(rows=r[1], states=r[2], eta=r[3], numerical=r[4], quarantined=r[5])
                      for r in con.execute('SELECT controller_uid,COUNT(*),COUNT(DISTINCT state_uid),'
                                           'COUNT(DISTINCT eta_uid),SUM(numerical_failure),SUM(conflict_quarantined) '
                                           'FROM rollout GROUP BY controller_uid')}
            row['controllers'] = []
            for uid, scene, config, quality in con.execute(
                    'SELECT c.controller_uid,s.name,c.config_json,c.compatibility_quality '
                    'FROM controller_config c JOIN scenario s USING(scenario_uid)'):
                row['controllers'].append(dict(uid=uid, scene=scene, config=json.loads(config),
                                               quality=quality, **counts.get(uid, {})))
            row['experiments'] = [dict(zip(('uid','name','path','protocol_hash','code_hash'), r)) for r in con.execute(
                'SELECT experiment_uid,name,path,protocol_hash,code_hash FROM experiment')]
        else:
            for table in tables:
                cols = [r[1] for r in con.execute('PRAGMA table_info('+table+')')]
                if table in ('rollout', 'continuation') and 'controller_uid' in cols:
                    fields = [c for c in ('controller_uid','chain','scenario','phase') if c in cols]
                    group = ','.join(fields)
                    extra = ',COUNT(DISTINCT state_uid)' if 'state_uid' in cols else ''
                    row[table] = [dict(zip(fields+['rows']+(['states'] if extra else []), r)) for r in con.execute(
                        'SELECT '+group+',COUNT(*)'+extra+' FROM '+table+' GROUP BY '+group)]
                elif table in ('probe_tasks','transfer_diagnostics','state_dependence_controls'):
                    row[table] = {'columns': cols, 'rows': con.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]}
        con.rollback()
        con.close()
        result['databases'].append(row)
        print(json.dumps(dict(database=str(p), tables=tables, controllers=len(row.get('controllers',[])))), flush=True)
    HERE.mkdir(exist_ok=True)
    (HERE/'cache_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print('inventory_saved', str(HERE/'cache_inventory.json'), flush=True)


if __name__ == '__main__':
    main()
