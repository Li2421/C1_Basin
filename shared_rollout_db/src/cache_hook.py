"""Lightweight pre-submit hook; never writes SQLite from workers."""
from __future__ import annotations
import json,os
from pathlib import Path
from .planner import preflight

def require_preflight(manifest_path,report_path=None):
    """Run before Slurm submission and reject avoidable exact duplicates."""
    report=preflight(manifest_path)
    if report_path:Path(report_path).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

def missing_only(manifest_path,output_path,report_path=None):
    obj=json.load(open(manifest_path));report=require_preflight(manifest_path,report_path);detail=report['details'];out=[]
    for req,d in zip(obj['requests'],detail):
        for seed in d.get('missing_seeds',[]):out.append({**req,'seed_keys':[seed]})
    Path(output_path).write_text(json.dumps({'source_manifest':str(manifest_path),'requests':out},indent=2)+'\n')
    return report

