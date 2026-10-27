"""Output encoding only; leave every numeric direction/action untouched."""
from core import clean,sha,HERE

def install():
    import runtime
    original_evaluate=runtime.ROUTE_EVALUATE
    original_run=runtime.run
    def evaluate(*args,**kwargs):
        columns,meta=original_evaluate(*args,**kwargs)
        return columns,clean(meta)
    def run(c):
        result,arrays=original_run(c)
        result['diagnostic_encoding_adapter']=dict(path=str(HERE/'recording_fix.py'),sha256=sha(HERE/'recording_fix.py'),scope='Convert NumPy diagnostic scalar types to JSON primitives; columns and control arithmetic unchanged.')
        return result,arrays
    runtime.ROUTE_EVALUATE=evaluate;runtime.run=run
