"""Adapt only the historical checkpoint field name in the checker."""
from common import *
import shutil

def main():
 src=HERE/'review_B.py';text=src.read_text();old="old['flow_checkpoint_sha256']==q['flow_checkpoint_sha256']";assert text.count(old)==1
 text=text.replace(old,"old.get('flow_checkpoint_sha256',old.get('checkpoint_sha256'))==q['flow_checkpoint_sha256'] and q['flow_checkpoint_sha256'] is not None")
 (HERE/'review_B_v2.py').write_text(text)
 src2=HERE/'B_final/analysis_worker.py';text=src2.read_text().replace("call('review_B.py')","call('review_B_v2.py')")
 out=HERE/'B_final_v2/analysis_worker.py';out.parent.mkdir(exist_ok=False);out.write_text(text)
 shutil.copy2(HERE/'B_FINALIZATION_ERROR.json',HERE/'B_FINALIZATION_ATTEMPT1_ERROR.json')
 seal('B_HISTORICAL_SCHEMA_CHECKER_FIX.receipt.json',[src,HERE/'review_B_v2.py',src2,out,HERE/'B_FINALIZATION_ATTEMPT1_ERROR.json'],reason='R10 raw uses checkpoint_sha256, R35/R37 raw uses flow_checkpoint_sha256; checker accepts the explicitly recorded alias and rejects missing checkpoint',execution_physics_labels_statistics_unchanged=True,no_new_rollouts=True)
 from single_job import submit
 submit('Bfinal2','B_final_v2/analysis_worker.py')
if __name__=='__main__':main()
