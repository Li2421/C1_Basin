from core import *
src=D/'axis3_cross_scene_v1/resource_monitor.py';s=src.read_text().replace("startswith('r34_')","startswith('r35_')").replace("['run.py','probes.py','advance.py','train.py']","['run.py','advance.py']")
(HERE/'resource_monitor.py').write_text(s)
p=HERE/'runtime.py';s=p.read_text().replace("source_outcome=old['outcome'],source_prefix_steps", "source_outcome=old['outcome'],source_evidence_grade=old.get('evidence_grade','DEVELOPMENT historical R10'),source_prefix_steps")
s=s.replace("newsteps==0 and source and source['source_experiment_uid']!=EXP", "newsteps==0 and source and (source['source_experiment_uid']!=EXP or source['source_evidence_grade'].startswith('DEVELOPMENT'))")
p.write_text(s)
