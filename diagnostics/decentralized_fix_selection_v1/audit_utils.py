from core import *
def source_inventory(name):
 files=set()
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and Path(p).suffix=='.py' and Path(p).is_file():files.add(str(Path(p).resolve()))
 write(HERE/name,dict(utc=utc(),imported_sources={p:sha(p) for p in sorted(files)}))
