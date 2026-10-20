from common import *
from factor_runtime import config_for,Config,LocalFrameConfig
import jax
jax.config.update('jax_enable_x64',False)
checks=[]
for n,v in read(HERE/'ORIGINAL_PROFILES.json').items():
 payload=v['payload']
 if n=='base_u_v10_local_macflow':base=read(HERE/'HISTORICAL_INVENTORY.json')['config'];cls=LocalFrameConfig
 else:base=read(payload['adapter_manifest_path'])['scenario_config'];cls=LocalFrameConfig if payload['observation_frame']=='local' else Config
 actual=config_for(n,700,.9).to_dict();original=cls(**base).to_dict();default=cls().to_dict();checks.append(dict(controller=n,original_equal=actual==original,default_equal=actual==default,field_differences={k:dict(factor=actual.get(k),original=original.get(k),default=default.get(k)) for k in set(actual)|set(original)|set(default) if actual.get(k)!=original.get(k) or actual.get(k)!=default.get(k)}))
passed=len(checks)==14 and all(q['original_equal'] and q['default_equal'] for q in checks);write(HERE/'CONFIG_UNIT_TEST.json',dict(utc=utc(),passed=passed,checks=checks,jax_enable_x64=bool(jax.config.x64_enabled),physical_steps=0));assert passed,checks
print('r.90 all14 original/default full configs PASS')
