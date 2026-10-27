"""Resumable unattended phase coordinator; raw numerical unknowns are final."""
from dispatch import *
import traceback

def main():
    if(HERE/'PROTECTION_VIOLATION.jsonl').exists():raise RuntimeError('HARD_STOP repository protection')
    if(HERE/'SCENE2_GEOMETRY_BLOCKER.json').exists()and not(HERE/'SCENE2_RESOLVED').exists():
        partial=read(HERE/'SCENES_FROZEN.json')
        assert partial.get('valid_subset_revision')==2,'Invalid type2 must first be explicitly removed from the frozen valid subset'
    while not(HERE/'SCENES_FROZEN.json').exists():
        pid=HERE/'calibration.pid'
        if pid.exists():
            try:os.kill(int(pid.read_text()),0)
            except ProcessLookupError:
                from calibrate import main as calibrate
                calibrate()
        else:
            from calibrate import main as calibrate
            calibrate()
        time.sleep(10)
    from validation import main as validation
    validation()
    try:
        from certify import main as certify
        certificate=certify()
    except Exception as exc:
        if(HERE/'PROTECTION_VIOLATION.jsonl').exists():raise
        certificate=dict(passed=False,enabled=[],error=repr(exc),traceback=traceback.format_exc(),policy='All optimizations disabled; original production path continues.')
        dump(HERE/'ACCELERATION_CERTIFICATE.json',certificate)
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n加速认证未完成，关闭四项优化后继续原生产路径：'+repr(exc)+'\n')
    import discovery
    if certificate.get('passed'):
        from acceleration import accelerated_case
        discovery.case=accelerated_case
    frozen=read(HERE/'SCENES_FROZEN.json')
    cohort,rows,minimum,dp=discovery.search(frozen)
    families=discovery.freeze_families(cohort,rows,minimum,dp)
    if(HERE/'PAPER_PARAM_DEVELOPMENT_PROTOCOL.json').exists():
        from paper_development import freeze_profiles
        freeze_profiles()
    discovery.holdout(frozen,families)
    from papers import main as papers
    papers()
    from analyze import main as analyze
    analyze()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(HERE/'COORDINATOR_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=datetime.datetime.now().astimezone().isoformat()))
        with(HERE/'GAPS.md').open('a')as f:f.write('\n协调进程待处理异常：'+repr(exc)+'。已完成轨迹和官方数据库记录保留，resume.sh 逐 UID 恢复。\n')
        raise
