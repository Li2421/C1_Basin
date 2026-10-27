"""Reuse an exact deterministic twin's diagnosis, never infer a new outcome."""
from experiment import *

CASE_SEED_KEYS={'seed','seed_key','rollout_uid'}
RAW_BOOKKEEPING={'case','rollout_uid','seed_key','future_index','timestamp','elapsed_seconds','new_physical_steps','deterministic_source'}
DIAGNOSTIC_METADATA={'utc','rollout_uid','source','trace_sha256','alias_diagnosis_reuse'}

def payload(diag):
    return {k:v for k,v in diag.items() if k not in DIAGNOSTIC_METADATA}

def reuse(raw):
    twin=raw.get('deterministic_source')
    if not isinstance(twin,dict) or 'path' not in twin or 'sha256' not in twin:return None
    path=Path(twin['path']).resolve()
    if not path.is_relative_to(HERE):
        if not path.is_relative_to(V1):return None
        from source_inputs import verify
        verify(path)
    if sha(path)!=twin['sha256']:return None
    original=read(path)
    if canonical({k:v for k,v in raw['case'].items() if k not in CASE_SEED_KEYS})!=canonical({k:v for k,v in original['case'].items() if k not in CASE_SEED_KEYS}):return None
    if canonical({k:v for k,v in raw.items() if k not in RAW_BOOKKEEPING})!=canonical({k:v for k,v in original.items() if k not in RAW_BOOKKEEPING}):return None
    if raw['trace_sha256']!=original['trace_sha256']:return None
    source=HERE/'DIAGNOSTICS'/f"{original['rollout_uid']}.json"
    if not source.exists():return None
    diag=read(source)
    if diag.get('trace_sha256')!=raw['trace_sha256']:return None
    provenance=diag.get('source',{})
    if provenance.get('sha256')!=twin['sha256'] or Path(provenance.get('path','')).resolve()!=path:return None
    return dict(result=payload(diag),provenance=dict(path=str(source),sha256=sha(source),rollout_uid=original['rollout_uid'],trace_sha256=raw['trace_sha256'],rule='Exact case except seed and exact complete raw except seed/bookkeeping; same saved physical trace and frozen diagnostic algorithm'))
