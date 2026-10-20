from core import *
source=D/'axis4a_geometry_horizon_v1/registry.py';s=source.read_text().replace("name='Round35 geometry horizon axis'","name='Round36 robot count axis'").replace("{'round':35}","{'round':36}").replace('2026100403','2026093011');(HERE/'registry.py').write_text(s);write(HERE/'REGISTRY_SOURCE.json',dict(source=str(source),sha256=sha(source),new_sha256=sha(HERE/'registry.py')))
