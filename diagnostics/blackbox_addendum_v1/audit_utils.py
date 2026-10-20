from core import *
def source_inventory(name):
    paths={str(Path(p).resolve()) for m in list(sys.modules.values()) if (p:=getattr(m,'__file__',None)) and Path(p).suffix=='.py' and Path(p).is_file()}
    write(HERE/name,dict(utc=utc(),sources={p:sha(p) for p in sorted(paths)}))
