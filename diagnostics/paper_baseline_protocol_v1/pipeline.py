"""Resumable coordinator for the ordered addendum and formal protocol."""
import os
os.environ['STB_COORDINATOR']='1'
import guard
from guard import HERE
import json,time,traceback,datetime,importlib

def record(stage,**fields):
    path=HERE/'PIPELINE_STATE.json';value=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=stage,**fields);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    from addendum_gates import native_methods
    assert(HERE/'PRIOR_TASK_DELIVERED.json').exists();native_methods()
    from port_training_stage import run
    record('matched_training_solo_return_repair');out=run()
    assert out['complete'],'Training before/after gate has infrastructure gaps'
    # Remaining phases are separate resumable stage modules; implementation can
    # be completed while this first numerical stage runs, without source edits.
    stages=[('synthetic_narrow_stage','run'),('consistent_geometry_stage','run'),('goal_alignment_stage','run'),('certify_execution','main'),('freeze_menus','main'),('tuning','main'),('freeze_analysis','main'),('formal_tests','main')]
    for module,entry in stages:
        record(module)
        getattr(importlib.reload(importlib.import_module(module)),entry)()
    record('mainline_numeric_complete_pending_report')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        record('error',error=repr(exc),traceback=traceback.format_exc())
        (HERE/'PIPELINE_ERROR.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),error=repr(exc),traceback=traceback.format_exc()),ensure_ascii=False,indent=2)+'\n')
        raise
