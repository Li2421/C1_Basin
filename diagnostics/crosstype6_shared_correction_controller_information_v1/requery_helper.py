"""Isolated CPU reference process, retaining original frozen-generator flags."""
from common import *
from requery_generator_v2 import FrozenMean

def main():
    mean=FrozenMean();print(json.dumps(dict(ready=True)),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line);result=mean.query(request['state_uid'],request['physical']);print(json.dumps(dict(ok=True,result=clean(result))),flush=True)
        except Exception as exc:print(json.dumps(dict(ok=False,error=type(exc).__name__+': '+str(exc))),flush=True)
if __name__=='__main__':main()
