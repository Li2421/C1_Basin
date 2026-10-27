"""Install a restart path without signalling the current root or official merger."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import subprocess,signal,time,ast


def command(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode().strip() if p.exists() else ''


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    ready='SEALED_MAIN_OPTIONAL_RESUME_READY_20261022.receipt.json';assert not (HERE/ready).exists()
    verify_receipt('MIGRATION_NUMBERS.receipt.json');verify_receipt('FINAL_DELIVERY_CLARITY_20261022.receipt.json')
    old=HERE/'scheduler_watchdog_spatial_mask_20261022.py';new=HERE/'scheduler_watchdog_sealed_main_resume_20261022.py';assert not new.exists();text=old.read_text()
    needle="    assert entry['coordinator']=='coordinator_spatial_mask_20261022.py',entry;verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')"
    replacement="    assert entry['coordinator'] in ('coordinator_spatial_mask_20261022.py','coordinator_sealed_main_resume_20261022.py'),entry;verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')\n    verify_receipt('"+ready+"');entry['coordinator']='coordinator_sealed_main_resume_20261022.py'"
    assert text.count(needle)==1;text=text.replace(needle,replacement)
    text=text.replace("entrypoint='scheduler_watchdog_spatial_mask_20261022.py'","entrypoint='scheduler_watchdog_sealed_main_resume_20261022.py'")
    ast.parse(text);new.write_text(text)
    for script in ('sealed_main_optional_resume_20261022.py','coordinator_sealed_main_resume_20261022.py'):ast.parse((HERE/script).read_text())
    seal(ready,[Path(__file__),new,HERE/'sealed_main_optional_resume_20261022.py',HERE/'coordinator_sealed_main_resume_20261022.py',HERE/'MIGRATION_NUMBERS.receipt.json',HERE/'FINAL_DELIVERY_CLARITY_20261022.receipt.json'],only_after_sealed_main=True,no_science_or_statistics_changes=True,no_repeated_main_work=True,official_merger_never_signalled=True)
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json');pre=dict(entry)
    oldchild=entry['downstream'];child_live=any(command(int(p.name))==sys.executable+' -B '+str(HERE/oldchild) for p in Path('/proc').iterdir() if p.name.isdigit())
    # Change a live child's configuration only on its next restart. A root
    # currently waiting in the merger has no live child and will read this
    # sealed target naturally when its existing merge action finishes.
    entry.update(utc=utc(),restart_coordinator='coordinator_sealed_main_resume_20261022.py',sealed_main_resume_receipt=ready,watchdog=new.name)
    if not child_live:entry.update(downstream='sealed_main_optional_resume_20261022.py',downstream_receipt=ready)
    watch=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'];expected=sys.executable+' -B '+str(HERE/pre['watchdog']);assert command(watch)==expected
    os.kill(watch,signal.SIGTERM)
    for _ in range(50):
        if command(watch)!=expected:break
        time.sleep(.1)
    write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
    with (HERE/'scheduler_watchdog_sealed_main_resume_20261022.log').open('a') as out:proc=subprocess.Popen([sys.executable,'-B',str(new)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'SEALED_MAIN_OPTIONAL_RESUME_INSTALL_20261022.json',dict(utc=utc(),previous_entry=pre,current_entry=entry,old_child_was_live=child_live,watchdog_pid=proc.pid,current_root_untouched=True,official_merger_untouched=True,scientific_worker_signals=0))
    seal('SEALED_MAIN_OPTIONAL_RESUME_INSTALL_20261022.receipt.json',[HERE/'SEALED_MAIN_OPTIONAL_RESUME_INSTALL_20261022.json',HERE/ready],scientific_worker_signals=0,merger_signals=0)
    print('Installed sealed-main restart path;current root,merger and science workers untouched',flush=True)


if __name__=='__main__':main()
