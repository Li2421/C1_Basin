"""Seal final appendices/index after the frozen numeric delivery succeeds."""
from common import *
import time,subprocess

def main():
 while not (HERE/'ADDITIONAL_VERIFICATION.receipt.json').exists():
  assert not (HERE/'FINALIZATION_ERROR.json').exists()
  assert not (HERE/'PIPELINE_ERROR.json').exists()
  time.sleep(10)
 r=read(HERE/'results.json');arms=read(HERE/'DESIGN_CONFIG.json')['arms'];desc={}
 for label,u in r['units'].items():
  if u['status']!='MAIN_COMPLETE':desc[label]=dict(status='GAP',reason='No capability-qualified full nine-arm main. Three-arm pilot cannot determine full-menu winning family.');continue
  a=u['analysis'];desc[label]=dict(status='DESCRIPTIVE',winning_arms=a.get('winning_arms'),schedule_minus_fixed=a.get('paired',{}).get('best_schedule_minus_fixed'),arm_order=arms,scope='Use crossfit and in-sample values from results, not a claim of a population-best family')
 write(HERE/'DESCRIPTIVE_ENDPOINTS.json',dict(utc=utc(),AC3=desc,AC4=r['five_layers']))
 text='# AC3 / AC4 描述性终点\n\n'
 for label,u in desc.items():text+=f"- {label}: {json.dumps(u,ensure_ascii=False)}\n"
 text+='\nAC4见FIVE_LAYER_EVIDENCE.md/json。能力门失败/无法判定的行未执行完整菜单，不能据三臂pilot选择完整九臂的最佳动作族。联合规划失败不是不可解证明，timeout不是deadlock。\n'
 (HERE/'ACTION_FAMILY_TABLE.md').write_text(text)
 subprocess.run([sys.executable,'-B',str(HERE/'five_round_index.py')],cwd=ROOT,check=True)
 seal('ALL_FIVE_DELIVERED.receipt.json',[HERE/'FIVE_ROUND_COMPLETION.receipt.json',HERE/'ADDITIONAL_VERIFICATION.receipt.json',HERE/'DESCRIPTIVE_ENDPOINTS.json',HERE/'ACTION_FAMILY_TABLE.md',HERE/'EXPERT_OVERLAP_FROZEN.receipt.json'],complete=True)
 print('All five sequential rounds delivered and final hashes verified',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'COMPLETION_INDEX_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
