import pathlib,time,json,os,signal,datetime
p=pathlib.Path(__file__).resolve().parent
while not(p/'SCENE2_RESOLVED').exists():
 q=p/'S2_execution.json'
 if q.exists() and json.loads(q.read_text()).get('complete'):
  pid=int((p/'main.pid').read_text());os.kill(pid,signal.SIGSTOP)
  (p/'SCENE2_COORDINATOR_PAUSED.json').write_text(json.dumps(dict(pid=pid,time=datetime.datetime.now().astimezone().isoformat(),after_stage='S2',before_stage='S3'))+'\n')
  break
 time.sleep(.1)
