"""Record explicit chat authorization and preserve the prior A/C delivery."""
from common import *
import shutil

def main():
 assert not (HERE/'PSTAR_DECISION.receipt.json').exists()
 target=HERE/'AC_DELIVERY_VERSION_1';target.mkdir(exist_ok=False)
 saved={}
 for f in sorted(HERE.iterdir()):
  if f.is_file():
   shutil.copy2(f,target/f.name);saved[str(f)]=dict(snapshot=str(target/f.name),sha256=sha(f));assert sha(target/f.name)==sha(f)
 write(HERE/'AC_DELIVERY_VERSION_1.manifest.json',dict(utc=utc(),files=saved,scope='All top-level files before B authorization; nested raw evidence remains immutable in original location'))
 seal('AC_DELIVERY_VERSION_1.receipt.json',[HERE/'AC_DELIVERY_VERSION_1.manifest.json',target/'NUMERICAL_COMPLETE.receipt.json',target/'SUPPLEMENTARY_DELIVERY.receipt.json'],supersession_policy='Original A/C completion receipts remain at their paths until explicitly versioned B finalization; old absolute references can be resolved through manifest snapshots')
 old=D/'axis4a_geometry_horizon_v1/PSTAR_CALIBRATION_CANDIDATES.json';v=read(old);assert v['observed500_count']==699
 line='**用户决定：** `p* 定义 = 候选 1`';now=utc()
 (HERE/'PSTAR_DECISION.md').write_text(line+'\n\n用户在对话中确认“是的”，确认上一条所述“候选1，仅用实际记录到第500步的699条，p*=0.9681620884830543，继续执行”。提前成功轨迹不补读、不做终点保持；提前unknown不补跑。原prompt文件只读，填写后的决定在本轮目录保存。\n')
 seal('PSTAR_DECISION.receipt.json',[HERE/'PSTAR_DECISION.md',old,HERE/'B_ANALYSIS_FROZEN.receipt.json'],selection='CANDIDATE_1',pstar=v['observed500_only_median'],authorized_rollouts=True,decision_line=line,line_sha256=hashlib.sha256(line.encode()).hexdigest(),utc_observed=now,filled_utc=now,filled_utc_basis='Explicit user chat authorization received before this record',user_confirmation='是的',confirmed_interpretation='Candidate 1: 699 actually observed p(500); continue B',source_prompt_sha256=sha('/home/zhihan/research/pending_prompts/prompt_round37_axis4a_addendum_20261016.md'),point_definition='Median over only 699 observed p500 trajectories; no terminal hold; 195 early successes and 2 pre500 unknown excluded from calibration only',observed500_count=699,before_B_submission=True)
 # Same resource scheduling, B-specific completion marker to survive phase gaps.
 src=HERE/'resource_monitor.py';text=src.read_text();assert text.count("HERE/'NUMERICAL_COMPLETE.receipt.json'")==1
 out=HERE/'resource_monitor_B.py';out.write_text(text.replace("HERE/'NUMERICAL_COMPLETE.receipt.json'","HERE/'B_FINAL_COMPLETE.receipt.json'"))
 seal('B_RESUMPTION_FROZEN.receipt.json',[HERE/'PSTAR_DECISION.receipt.json',HERE/'AC_DELIVERY_VERSION_1.receipt.json',src,out,HERE/'B_driver.py'],resource_change='Completion marker only; scheduling policy unchanged',no_A_C_reruns=True)
 print('Candidate1 frozen',v['observed500_only_median'],flush=True)
if __name__=='__main__':main()
