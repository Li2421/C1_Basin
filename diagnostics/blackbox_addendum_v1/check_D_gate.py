from common import *
from audit_utils import source_inventory

def main():
    checks=[]
    for c in read(HERE/'replay_D_CASES.json'):
        r=read(HERE/'replay_D/raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);assert sha(c['reference_path'])==c['reference_sha256'];z=np.load(r['trace_path']);ref=np.load(old['trace_path']);a=z['positions'];b=ref['positions'];fa=r['first_action'];fb=old['first_action'];q=dict(group=c['replay_group'],controller=c['base_controller_name'],first_action_bytes=(fa is None and fb is None) or (fa is not None and fb is not None and np.asarray(fa,np.float64).tobytes()==np.asarray(fb,np.float64).tobytes()),positions_bytes=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),termination=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length'],never_active=not z['rhr_active'].any(),jax_enable_x64=r['jax_enable_x64']);q['passed']=all(q[k] for k in ['first_action_bytes','positions_bytes','termination','steps','never_active']);checks.append(q)
    out=dict(utc=utc(),Flow_inactive_passed=all(q['passed'] for q in checks if q['group']=='Flow_inactive'),Novel_inactive_passed=all(q['passed'] for q in checks if q['group']=='Novel_inactive'),trigger_passed=read(HERE/'TRIGGER_TESTS.json')['passed'],checks=checks);write(HERE/'D_REPRODUCTION_GATE.json',out);seal('D_REPRODUCTION_GATE.receipt.json',[HERE/'D_REPRODUCTION_GATE.json',HERE/'TRIGGER_TESTS.json'],before_new_tuning_labels=True);source_inventory('imports_check_D_gate.json');print('Right-hand gates',out['Flow_inactive_passed'],out['Novel_inactive_passed'])
if __name__=='__main__':main()
