from common import *
p=HERE/'runtime.py';s=p.read_text().replace(" s=next(x for x in p['states']", " v=p['profiles'][c['profile_key']]['payload'];assert all(sha(HERE/k)==h for k,h in v['execution_source_sha256'].items())\n s=next(x for x in p['states']")
p.write_text(s)
archive=HERE/'gate_pre_execution_design';archive.mkdir(exist_ok=True)
for path in list(HERE.glob('gate_*')):
 if path.is_file():path.rename(archive/path.name)
write(archive/'AMENDMENT.json',dict(utc=utc(),reason='Before execution: propagate source evidence grade through exact derived prefixes and assert execution hashes. Superseded requests never executed. Existing identity rows remain append-only.',pending_job='21833'))
