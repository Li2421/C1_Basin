from common import *
from shared_rollout_db.src.rollout_db import uid,canonical
# The official merger derives logical future_index alone. Preserve immutable raw
# evidence and existing DB rows; map earlier request aliases to actual exact UIDs.
p=HERE/'design.py';s=p.read_text().replace("canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']))","canonical(dict(future_index=seed))");p.write_text(s)
p=HERE/'factor_design.py';s=p.read_text().replace("canonical(dict(expert_index=0,deterministic=True,rng_namespace=s['rng_namespace']))","canonical(dict(future_index=0))");p.write_text(s)
p=HERE/'advance.py';s=p.read_text().replace("rows=read(HERE/f'{stage}_CASES.json');checks=[]", "rows=read(HERE/f'{stage}_CASES.json');checks=[];aliases=[]\n from shared_rollout_db.src.rollout_db import uid,canonical")
s=s.replace("   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None,c['rollout_uid'];", "   actual=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=c['seed']))));aliases.append(dict(request_rollout_uid=c['rollout_uid'],db_rollout_uid=actual));row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(actual,)).fetchone();assert row is not None,actual;")
s=s.replace(" write(HERE/f'{stage}_POSTFLIGHT.json',", " write(HERE/f'{stage}_DB_IDENTITY_MAP.json',aliases)\n write(HERE/f'{stage}_POSTFLIGHT.json',")
p.write_text(s)
write(HERE/'SEED_TRANSPORT_REPAIR.json',dict(utc=utc(),reason='Official historical_ingest.seed normalizes future_index alone, ignoring redundant seed_key root/namespace. Initial gate/expert_gate/baseline planned UIDs had redundant seed fields. Journals successfully imported exact compatible labels under canonical logical-index UIDs. No missing physical evidence and no repeat simulation.',before_next_horizon_design=True,new_design_seed_key='canonical({future_index:seed})',preserved=['original requests/preflight/raw/traces/journals/DB rows'],postflight='Explicit request-to-DB identity maps; verify raw_record_hash, flags, physical state and exact controller; no UPDATE/DELETE or duplicate journal'))
