import bootstrap
from experiment import *
from audit_supplement import evidence
from functools import lru_cache

@lru_cache(None)
def sources_index():
    out={}
    # Registry UIDs are filenames in immutable artifact directories. Index only
    # physical raw artifacts; replay manifests and MISSING_CASES are not evidence.
    # This also avoids parsing the 872 MB capability manifest for every lookup.
    for p in HERE.glob('*/raw/roll_*.json'):
        rid=p.stem
        assert rid not in out or Path(out[rid]).resolve()==p.resolve(),('Duplicate local raw identity',rid,out.get(rid),str(p))
        out[rid]=str(p)
    return out

def locate(stage,c):
    p=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
    if p.exists():return p
    if c.get('reuse_source'):return Path(c['reuse_source'])
    p=sources_index().get(c['rollout_uid'])
    assert p,('Missing immutable raw for exact DB reuse',stage,c['rollout_uid']);return Path(p)
def load(stage,c):
    p=locate(stage,c);r=read(p);assert r['rollout_uid']==c['rollout_uid'];return r
