"""Observe explicit filling of the user-owned decision field; never choose a default."""
from common import *
import re,time,subprocess,shlex
PROMPT=Path('/home/zhihan/research/pending_prompts/prompt_round37_axis4a_addendum_20261016.md')
def main():
 while not (HERE/'NUMERICAL_COMPLETE.receipt.json').exists():
  if (HERE/'PSTAR_DECISION.receipt.json').exists():return
  text=PROMPT.read_text();line=next(x for x in text.splitlines() if x.startswith('**用户决定：**'))
  if '待填写' not in line:
   choices=[k for k in [1,2,3] if re.search(r'候选\s*'+str(k),line)]+(['ABANDON'] if '放弃' in line else [])
   if len(choices)!=1:
    write(HERE/'PSTAR_DECISION_NEEDS_DEFINITION.json',dict(utc=utc(),line=line,reason='Exactlyoneexplicitchoice required; neverdefault fromunfilled alternatives'));time.sleep(15);continue
   if '放弃 S-prog' in line or '放弃S-prog' in line:selection='ABANDON';value=None
   elif re.search(r'候选\s*1',line):selection='CANDIDATE_1';value=read(D/'axis4a_geometry_horizon_v1/PSTAR_CALIBRATION_CANDIDATES.json')['observed500_only_median']
   elif re.search(r'候选\s*2',line):selection='CANDIDATE_2';value=read(D/'axis4a_geometry_horizon_v1/PSTAR_CALIBRATION_CANDIDATES.json')['known_plus_success_hold_median']
   else:
    write(HERE/'PSTAR_DECISION_NEEDS_DEFINITION.json',dict(utc=utc(),line=line,reason='Customdefinition requires exact offline specification; noBrollout'));time.sleep(15);continue
   now=utc();(HERE/'PSTAR_DECISION.md').write_text(line+'\n');seal('PSTAR_DECISION.receipt.json',[HERE/'PSTAR_DECISION.md'],selection=selection,pstar=value,authorized_rollouts=value is not None,decision_line=line,line_sha256=hashlib.sha256(line.encode()).hexdigest(),utc_observed=now,filled_utc=datetime.fromtimestamp(PROMPT.stat().st_mtime,timezone.utc).isoformat(),filled_utc_basis='User-owned sourcefile mtime; observationtime separatelyrecorded',source_prompt_sha256=sha(PROMPT),candidate2_interval=read(D/'axis4a_geometry_horizon_v1/PSTAR_CALIBRATION_CANDIDATES.json')['full_median_interval_given_success_hold'] if selection=='CANDIDATE_2' else None,point_definition='Exact existing frozen candidatepoint; interval remains anidentificationbound, not anobserved full896median')
   print('Explicituserdecision observed',selection,value,flush=True);return
  time.sleep(15)
if __name__=='__main__':main()
