"""Frozen six full paper-mechanism ports; same plant/maps/horizon, native safety."""
from dispatch import *

OLD=ROOT/'diagnostics/paper_baselines_20261028'

def prepare():
    ports=HERE/'ports';ports.mkdir(exist_ok=True);manifest=[]
    for name in ('paper_adapters.py','mgr_geometry.py','impc_adapter.py','formula_baselines.py'):
        source=OLD/name;data=source.read_text()
        adapted=data.replace('HERE=Path(__file__).resolve().parent',f'HERE=Path({str(OLD)!r})')
        target=ports/name
        if target.exists():assert target.read_text()==adapted
        else:target.write_text(adapted)
        manifest.append(dict(source=str(source),source_sha256=sha(source),target=str(target),target_sha256=sha(target),change='Only HERE redirected to the sealed read-only vendor/dependency root. Mechanism code unchanged.'))
    dump(HERE/'PAPER_PORT_INPUTS.json',dict(files=manifest,pins=read(OLD/'SOURCE_PINS.json'),options=read(OLD/'paper/FINAL_FROZEN.json')['selected'],scope='Same physical scenes, radius, wall margin, dt, speed, arrival and horizon. Full published mechanisms use their own controller and safety solver as in the sealed previous round. These are explicitly labelled full-mechanism common-plant ports, not nominal-controlled ablations or exact native paper benchmark reproductions. Their internal planning/temporary-goal logic sees true terminal goals; external route guidance belongs to the frozen C1 nominal pipeline.'))

def main():
    prepare();assert(HERE/'FAMILIES_FROZEN.json').exists()
    frozen=read(HERE/'SCENES_FROZEN.json');options=read(OLD/'paper/FINAL_FROZEN.json')['selected'];cs=[]
    for e in frozen['entries']:
        if 'scene'not in e:continue
        s=e['scene']
        for i in range(4,8):
            for method,opts in options.items():
                c=case(s,controllers()[0],i,purpose='paper_holdout')
                c['controller_payload'].update(nominal=dict(type_code=method),paper_method=method,paper_options=opts,safety_layer='paper_own_controller_safety',paper_port_inputs_sha256=sha(HERE/'PAPER_PORT_INPUTS.json'),paper_worker_sha256=sha(HERE/'paper_episode.py'))
                c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
                c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));cs.append(c)
    execute(cs,'paper_holdout',worker_entry='worker_papers.py')

if __name__=='__main__':main()
