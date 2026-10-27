"""Read-only foreground observation of final closure milestones."""
from experiment import *


def main():
    guard = read(HERE/'LOCAL_CPU_GUARD_STATUS.json')
    if isinstance(guard.get('local_cpu'), int):
        os.sched_setaffinity(0, {guard['local_cpu']})
    names = ('OPTIONAL_NUMBERS.receipt.json', 'DB_POSTFLIGHT.receipt.json',
             'FIGURES.receipt.json', 'INTEGRITY_END.receipt.json',
             'FINAL_REPORTS.receipt.json', 'V2_COMPLETE.json')
    milestones = {n: (HERE/n).exists() for n in names}
    issues = [str(p.relative_to(HERE))
              for pattern in ('COORDINATOR_ACTION_ERROR.json',
                              '*/NEEDS_IMPLEMENTATION_REVIEW.json',
                              'OPTIONAL_GAPS.json') for p in HERE.glob(pattern)]
    active = []
    wanted = {'optional_results_memo_20261022.py', 'db_merge.py',
              'postflight_correction_closed_scope_20261023_quota18_now_20261022.py',
              'plot_training_selected_primary_20261022.py', 'integrity.py',
              'final_reports_protocol_order_20261022.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            selected = [Path(os.fsdecode(v)) for v in argv
                        if v.startswith(os.fsencode(str(HERE)+'/'))]
            selected = [p.name for p in selected if p.name in wanted]
            if not selected:
                continue
            stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            active.append(dict(pid=int(proc.name), entry=selected[0],
                               cpu_seconds=round((int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'), 2),
                               resident_MiB=round(int(stat[21])*os.sysconf('SC_PAGE_SIZE')/2**20, 1)))
        except (OSError, ValueError, IndexError):
            pass
    logs = sorted((HERE/'coordinator_actions').glob('*.log'),
                  key=lambda p:p.stat().st_mtime, reverse=True)[:3]
    result = dict(utc=utc(), milestones=milestones, issues=issues,
                  active=active, recent_actions=[dict(name=p.name, bytes=p.stat().st_size) for p in logs])
    print(canonical(result), flush=True)
    return result


if __name__ == '__main__':
    main()
