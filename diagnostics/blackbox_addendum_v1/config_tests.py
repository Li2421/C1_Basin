from common import *
from factor_runtime import config_for,make
from dataclasses import asdict
from audit_utils import source_inventory
import jax
jax.config.update('jax_enable_x64',False)
def main():
    rows=[]
    for name in read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']:
        rt=make(name,700,.9,{});new=config_for(name,700,.9);default=type(new)();physical_keys=['agent_collision_margin','agent_radius','collision_margin','dt','goal_tolerance','max_speed','max_steps','obstacle_radius','outer_radius','terminate_on_collision','terminate_on_success'];diff={k:[getattr(new,k),getattr(default,k)] for k in asdict(new) if getattr(new,k)!=getattr(default,k)};same=asdict(new)==asdict(rt.config);rows.append(dict(controller=name,original_runtime_config_equal=same,default_config_equal=not diff,default_differences=diff,physical_defaults_equal=all(getattr(new,k)==getattr(default,k) for k in physical_keys),config=asdict(new)))
    passed=all(r['original_runtime_config_equal'] and r['default_config_equal'] for r in rows);write(HERE/'RADIUS_CONFIG_TEST.json',dict(utc=utc(),passed=passed,checks=rows,new_physical_steps=0,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory('imports_config_tests.json');print('Radius baseline configs pass',passed)
if __name__=='__main__':main()
