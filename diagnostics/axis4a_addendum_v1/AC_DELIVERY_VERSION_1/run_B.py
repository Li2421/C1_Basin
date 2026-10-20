from common import *
assert sys.argv[1]=='B'
decision=read(HERE/'PSTAR_DECISION.receipt.json');assert decision['authorized_rollouts'] and decision['pstar'] is not None
assert (HERE/'SELECTED_PROGRESS_TEST.receipt.json').exists() and (HERE/'B_ANALYSIS_FROZEN.receipt.json').exists()
for c in read(HERE/'B_MISSING_CASES.json'):assert c['arm']=='Sprog' and c['progress_threshold']==decision['pstar']
from run import main
if __name__=='__main__':main()
