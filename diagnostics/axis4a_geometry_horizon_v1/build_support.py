from core import *
# Adapt identity registration only inside current round; original hash retained.
src=D/'axis3_cross_scene_v1/registry.py'
s=src.read_text().replace("name='Round34 cross-scene axis'","name='Round35 geometry horizon axis'").replace("{'round':34}","{'round':35}")
s=s.replace("  for s in p['states']:\n", "  for v in p['scenarios'].values():ins('scenario',v)\n  for s in p['states']:\n")
s=s.replace("p['environments'][s['scenario']]['config']['config']","s['evaluation_config']")
(HERE/'registry.py').write_text(s)
write(HERE/'SUPPORT_COPY_PROVENANCE.json',dict(registry_source=str(src),registry_source_sha256=sha(src),core_source=str(D/'axis3_cross_scene_v1/core.py'),core_source_sha256=sha(D/'axis3_cross_scene_v1/core.py'),audit_source=str(D/'axis3_cross_scene_v1/audit.py'),audit_source_sha256=sha(D/'axis3_cross_scene_v1/audit.py')))
