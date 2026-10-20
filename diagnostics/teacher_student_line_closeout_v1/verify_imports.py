from core import *
from ring_exchange.expert import circulation_signature
precision()
files={}
for module in list(sys.modules.values()):
 p=getattr(module,'__file__',None)
 if p and pathlib.Path(p).suffix=='.py' and pathlib.Path(p).is_file() and pathlib.Path(p).resolve().is_relative_to(ROOT) and not pathlib.Path(p).resolve().is_relative_to(HERE):files[str(pathlib.Path(p).resolve())]=sha(p)
a=read(HERE/'INTEGRITY_START.json');write(HERE/'IMPORT_VERIFICATION.json',dict(utc=utc(),read_only_project_sources=files,start_hash_crosschecks={p:h==a['files'][p] for p,h in files.items() if p in a['files']},project_imports_not_in_explicit_start_source_list=[p for p in files if p not in a['files']],copied_old_modules=[],precision=precision(),simulator_or_planner_called=False))
print('read-only import sources verified',len(files))
