from common import *
import subprocess,time

def call(script,*args):
 with (HERE/(script.replace('.py','')+'_FINAL.log')).open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 assert (HERE/'ALL_LABELS_COMPLETE.receipt.json').exists()
 while not (HERE/'MEMORY_MONITOR_FINISHED.receipt.json').exists():time.sleep(2)
 call('analyze.py');call('performance.py');call('audit.py','END');call('delivery.py');call('review.py')
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
