"""Unique temporary names for concurrently shared derived evidence caches."""
from experiment import HERE,clean,json,Path,os
import uuid

def write(path,value):
    path=Path(path);assert path.resolve().is_relative_to(HERE),path
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        stream.write(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
