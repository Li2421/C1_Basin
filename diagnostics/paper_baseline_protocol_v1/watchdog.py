"""Administrative recovery with bounded restarts; never manufactures results."""
import guard
from guard import HERE
import subprocess,time,json,datetime
from resource_watch import enforce,jobs
from progress_report import update


def main():
    path=HERE/'WATCHDOG_STATE.json'
    value=json.loads(path.read_text())if path.exists()else dict(restarts=0)
    while True:
        try:
            enforce();update()
            live=jobs();coordinator=any(r['name']=='pbp1_pipeline'for r in live)
            if(HERE/'COMPLETE.json').exists()and not any(r['ours']for r in live):
                now=datetime.datetime.now().astimezone().isoformat()
                delivered=json.loads((HERE/'COMPLETE.json').read_text())
                (HERE/'PIPELINE_STATE.json').write_text(json.dumps(dict(time=now,stage='delivered',
                    main_numeric_work_complete=delivered.get('main_numeric_work_complete'),complete=delivered.get('complete'),
                    receipt='COMPLETE.json'),ensure_ascii=False,indent=2)+'\n')
                (HERE/'TASK_CPU_RELEASED').write_text(now+'\n')
                break
            if not coordinator:
                state=json.loads((HERE/'PIPELINE_STATE.json').read_text())if(HERE/'PIPELINE_STATE.json').exists()else{}
                if state.get('stage')=='error'and value['restarts']<3:
                    # Bounded infrastructure recovery; an implementation error
                    # remains visible and requires the active agent's repair.
                    error=state.get('error','')
                    if any(word in error for word in ('Slurm','subprocess','infrastructure','temporarily unavailable')):
                        with(HERE/'pipeline.log').open('ab')as log:
                            subprocess.Popen(['bash',str(HERE/'resume.sh')],stdout=log,stderr=log,start_new_session=True)
                        value['restarts']+=1
            value.update(time=datetime.datetime.now().astimezone().isoformat(),observed_coordinator=coordinator)
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        except Exception as exc:
            with(HERE/'watchdog_events.jsonl').open('a')as out:out.write(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),error=repr(exc)))+'\n')
        time.sleep(30)


if __name__=='__main__':main()
