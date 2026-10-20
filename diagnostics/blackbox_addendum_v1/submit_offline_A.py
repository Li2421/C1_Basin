from common import *
import subprocess,time,re,traceback

def main():
 write(HERE/'A_STATS_PENDING.json',dict(utc=utc(),status='WAITING_FOR_CPU',new_rollouts=0))
 while True:
  queue=subprocess.check_output(['squeue','-r','-h','-o','%i|%u|%j|%T|%C|%b'],text=True);rows=[x.split('|')for x in queue.splitlines()];used=sum(int(x[4])for x in rows if x[3]in ['RUNNING','COMPLETING','CONFIGURING'])
  if used<18:break
  time.sleep(5)
 seal('A_STANDARD_PRESENTATION_FROZEN.receipt.json',[HERE/'standard_A.py',HERE/'geometry_curves.py',HERE/'standard_A.sbatch'],before_standard_table_and_curve_computation=True,new_rollouts=0)
 r=subprocess.check_output(['sbatch',str(HERE/'standard_A.sbatch')],text=True);job=re.search(r'\d+',r).group();write(HERE/'A_STATS_JOB.json',dict(utc=utc(),job=job,response=r,queue_before=rows,active_CPUs_before=used,requested_CPUs=1,CPU_cap=18,mode='Offline source-label statistics and curves; no simulation or DB writes. Governor treats this nonworker job as a reserved CPU.'));print(r,flush=True)
 while subprocess.check_output(['squeue','-r','-h','-j',job],text=True).strip():time.sleep(5)
 assert (HERE/'A_STANDARD_RESULTS.receipt.json').exists()and(HERE/'A_CONTROLLER_RADIUS_CURVES.receipt.json').exists();write(HERE/'A_STATS_READY.json',dict(utc=utc(),job=job,complete=True))
if __name__=='__main__':
 try:main()
 except Exception:write(HERE/'A_STATS_ERROR.json',dict(utc=utc(),error=traceback.format_exc()));raise
