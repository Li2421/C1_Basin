"""Record actual imported source files and runtime configuration."""
import inspect
from local_probes import make_runtime,save,sha
import single_integrator.cbf
import shared_control.hard_projection
import utils.networks
import jax

rows=[]
for sc in ['toy_give_way','ring_exchange','double_bottleneck','four_way_intersection']:
    rt=make_runtime(sc)
    files=[inspect.getfile(type(rt)),inspect.getfile(type(rt.agent)),inspect.getfile(utils.networks.ActorVectorField),
           inspect.getfile(single_integrator.cbf),inspect.getfile(shared_control.hard_projection)]
    rows.append(dict(scene=sc,config=rt.config.to_dict(),agent_config=dict(rt.agent.config),cbf=rt.cbf.to_dict(),
                     scale=rt.scale,mean=rt.mean,x64=bool(jax.config.x64_enabled),sources={s:sha(s) for s in files}))
save('runtime_inspection.json',rows)
