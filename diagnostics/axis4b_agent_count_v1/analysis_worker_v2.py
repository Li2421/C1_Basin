"""Original frozen estimators/review, plus explicit evidence-derivation verification."""
from common import *
import subprocess,time

def call(script,*args):
 with (HERE/(script.replace('.py','')+'_FINAL.log')).open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 assert (HERE/'ALL_LABELS_COMPLETE.receipt.json').exists()
 while not (HERE/'MEMORY_MONITOR_FINISHED.receipt.json').exists():time.sleep(2)
 call('analyze.py');call('performance_v2.py');call('prefix_audit.py');call('audit.py','END');call('delivery.py');call('review.py')
 seal('ADDITIONAL_VERIFICATION.receipt.json',[HERE/'NUMERICAL_COMPLETE.receipt.json',HERE/'EXACT_PREFIX_REUSE_AUDIT.json',HERE/'EXACT_REUSE_IMPLEMENTATION_FROZEN.receipt.json',HERE/'REUSE_QUALIFICATION.md'],passed=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
