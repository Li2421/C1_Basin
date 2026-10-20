"""Render frozen-rule representative episodes from verified simulation traces."""
from common import *
from html import escape

COLORS=('#1768ac','#d95d39','#237b4b','#854d9b')


def plot(record,config,title,path):
    assert sha(record['trace_path'])==record['trace_sha256']
    with np.load(record['trace_path'],allow_pickle=False) as z:positions=z['positions'];goals=z['goals']
    radius=config['outer_radius'];scale=580/(2*(radius+.25))
    xy=lambda a:(360+float(a[0])*scale,390-float(a[1])*scale)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="790" viewBox="0 0 720 790">',
         '<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:DejaVu Sans,Arial,sans-serif}</style>']
    def text(x,y,value,size=14):svg.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')
    text(30,30,title,18);text(30,55,f'Outcome: {record["outcome"]}; recorded steps: {len(positions)-1}')
    svg.append(f'<circle cx="360" cy="390" r="{radius*scale}" fill="#f8fafb" stroke="#333" stroke-width="2"/>')
    svg.append(f'<circle cx="360" cy="390" r="{config["obstacle_radius"]*scale}" fill="#c8ccd1" stroke="#555"/>')
    for i,color in enumerate(COLORS):
        pts=' '.join(f'{xy(row)[0]:.3f},{xy(row)[1]:.3f}' for row in positions[:,i])
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for position,filled in ((positions[0,i],False),(positions[-1,i],True)):
            x,y=xy(position);svg.append(f'<circle cx="{x}" cy="{y}" r="{config["agent_radius"]*scale}" fill="{color if filled else "white"}" fill-opacity=".8" stroke="{color}" stroke-width="2"/>')
        x,y=xy(goals[i]);svg.append(f'<path d="M {x-6} {y-6} L {x+6} {y+6} M {x-6} {y+6} L {x+6} {y-6}" stroke="{color}" stroke-width="2.5"/>')
    text(30,735,'Open circles: start; filled circles: final; crosses: goals. Robot radius to scale.',13)
    text(30,760,'Recorded trajectories only. Timeout is not a verified deadlock.',13)
    svg.append('</svg>');path.write_text('\n'.join(svg)+'\n')


def main(phase):
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists()
    for path,digest in read(HERE/'FROZEN_PIPELINE.receipt.json')['files'].items():assert sha(path)==digest
    p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json')
    details=read(HERE/'LOCKED_CONTROLLER_DETAILS.json');chosen={r['controller']:r['pipeline_arm'] for r in details}
    design=read(HERE/'TRAJECTORY_FIGURE_DESIGN.json')
    assert design['utc']<read(HERE/'LOCKED_CHOICES.receipt.json')['utc']
    selection=[];folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for t in sorted({r['type'] for r in details}):
        pool=[r for r in details if r['type']==t];median=float(np.median([r['success_pct'] for r in pool]))
        controller=min(pool,key=lambda r:(abs(r['success_pct']-median),r['controller']))['controller']
        selected_cases=[c for c in cases if c['base_controller_name']==controller and c['arm']==chosen[controller]]
        records=[read(HERE/phase/'raw'/f"{c['rollout_uid']}.json") for c in selected_cases]
        median_length=float(np.median([r['episode_length'] for r in records]))
        idx=min(range(len(records)),key=lambda i:(abs(records[i]['episode_length']-median_length),selected_cases[i]['state_index'],selected_cases[i]['seed']))
        state_index=selected_cases[idx]['state_index'];seed=selected_cases[idx]['seed']
        config=p['native_manifest']['scenario_config'];outputs=[]
        for label,arm in [('nominal_only','nominal_only'),('safety_baseline','safety_baseline'),('pipeline',chosen[controller])]:
            case=next(c for c in cases if c['base_controller_name']==controller and c['state_index']==state_index and c['seed']==seed and c['arm']==arm)
            raw=HERE/phase/'raw'/f"{case['rollout_uid']}.json";record=read(raw)
            image=folder/f'{t}_{label}_trajectory.svg'
            plot(record,config,f'{controller}: {label} ({arm})',image)
            outputs.append(dict(method=label,arm=arm,raw_path=str(raw),raw_sha256=sha(raw),trace_path=record['trace_path'],
                trace_sha256=record['trace_sha256'],image=str(image),image_sha256=sha(image)))
        selection.append(dict(type=t,controller=controller,state_index=state_index,seed=seed,
            selection_rule=design['rule'],median_controller_success_pct=median,median_selected_episode_length=median_length,outputs=outputs))
    write(HERE/'LOCKED_TRAJECTORY_FIGURES.json',dict(utc=utc(),selections=selection,simulation_modified=False,new_rollouts=0))
    seal('LOCKED_TRAJECTORY_FIGURES.receipt.json',[HERE/'LOCKED_TRAJECTORY_FIGURES.json',HERE/'plot_locked_episodes.py',HERE/'TRAJECTORY_FIGURE_DESIGN.json'])
    print('Representative locked trajectory figures rendered from verified traces')


if __name__=='__main__':main(sys.argv[1])
