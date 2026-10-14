"""Read-only fingerprints of inherited simulator/controller assets."""
from core import *

def main():
    prereg();p=read(HERE/'protocol.json');field=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004');checks={}
    def check(path,expected):
        path=Path(path);actual=sha(path);assert actual==expected,('Inherited asset changed',str(path));checks[str(path)]=dict(expected=expected,observed=actual)
    check(R9/'run.py',p['runner_sha256']);check(HERE/'run.py',p['io_binding_sha256']);check(R9/'core.py',p['original_round9_core_sha256']);check(HERE/'core.py',p['core_sha256']);check(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py',p['flow_adapter_sha256'])
    check(field/'source/ring_exchange/environment.py',p['environments']['ring_exchange']['config']['environment_code_sha256'])
    for profile in p['base_profiles'].values():
        v=profile['payload'];check(v['flow_checkpoint_path'],v['flow_checkpoint_sha256']);check(v['execution_callable'],v['execution_callable_sha256'])
        if v.get('adapter_manifest_path'):check(v['adapter_manifest_path'],v['adapter_manifest_sha256'])
        for name,value in v['action_code_hashes'].items():check(field/name,value)
        check(field/'source/shared_control/hard_projection.py',v['safety_projection_sha256'])
    write(HERE/'INHERITED_ASSET_AUDIT.json',dict(utc=utc(),passed=True,checks=checks));print('Inherited asset audit PASS',len(checks))
if __name__=='__main__':main()
