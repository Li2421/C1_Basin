from common import *
from audit_utils import source_inventory

def main():
    checks=[]
    for c in read(HERE/'replay_A_CASES.json'):
        path=HERE/'replay_A/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(c['reference_path'])==c['reference_sha256'];old=read(c['reference_path']);assert sha(old['trace_path'])==old['trace_sha256'];a=np.load(r['trace_path'])['positions'];b=np.load(old['trace_path'])['positions'];fa=r['first_action'];fb=old['first_action'];q=dict(group=c['replay_group'],controller=c['base_controller_name'],radius=c['r'],first_action_bytes=(fa is None and fb is None) or (fa is not None and fb is not None and np.asarray(fa,np.float64).tobytes()==np.asarray(fb,np.float64).tobytes()),positions_bytes=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),termination=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length'],reference=c['reference_path'],jax_enable_x64=r['jax_enable_x64']);q['passed']=all(q[k] for k in ['first_action_bytes','positions_bytes','termination','steps']);checks.append(q)
    passed=len(checks)==32 and all(q['passed'] for q in checks) and read(HERE/'RADIUS_CONFIG_TEST.json')['passed'];write(HERE/'A_REPRODUCTION_GATE.json',dict(utc=utc(),passed=passed,checks=checks,config_default_bitwise_compatible=read(HERE/'RADIUS_CONFIG_TEST.json')['passed']));seal('A_REPRODUCTION_GATE.receipt.json',[HERE/'A_REPRODUCTION_GATE.json',HERE/'RADIUS_CONFIG_TEST.json'],before_A1_A2_labels=True);source_inventory('imports_replay_check.json');print('A reproduction gate',passed)
if __name__=='__main__':main()
