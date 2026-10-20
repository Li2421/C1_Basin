from common import *
import subprocess

def call(script,*args):
 with (HERE/(script.replace('.py','')+'_FINAL.log')).open('w') as f:subprocess.run([sys.executable,'-B',str(HERE/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 assert (HERE/'ALL_FACTOR_LABELS_COMPLETE.receipt.json').exists();call('analyze.py');call('complete_experts.py');call('audit.py','END');call('delivery_qualified.py');report=HERE/'REPORT.md';report.write_text(report.read_text()+(HERE/'DOCUMENTATION_SUPPLEMENT.md').read_text()+'\n'+(HERE/'EXPERT_SCOPE_CLARIFICATION.md').read_text());call('review.py')
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
