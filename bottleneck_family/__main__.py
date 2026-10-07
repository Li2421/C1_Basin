"""Generate and inspect geometry only; this command does not evaluate a policy."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from html import escape

from .scenario import Config, build_instance


def preset(name, count, seed=0):
    base = Config(num_agents=count, seed=seed)
    if name == 'gap':
        return base
    if name == 'double':
        return replace(base, barrier_x=(-1.5,1.5), openings=(((0.,.5),),((0.,.5),)))
    if name == 'offset':
        return replace(base, barrier_x=(-1.5,1.5), openings=(((-.7,.5),),((.7,.5),)))
    if name == 'wide':
        return replace(base, openings=(((0.,1.2),),))
    if name == 'three_doors':
        return replace(base, openings=(((-2.,.5),(0.,.5),(2.,.5)),))
    if name == 'clutter':
        return replace(base, extra_rectangles=((-3.,-.6,-2.,.6),(2.,1.,3.,2.)))
    raise ValueError(name)


def svg(instance, title=None, width=1000, height=540):
    c = instance.config
    scale = min((width-60)/(2*c.half_length),(height-110)/(2*c.half_height))
    def xy(p):
        return width/2+p[0]*scale, height/2+20-p[1]*scale
    body = [f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="24" y="25" font-family="sans-serif" font-size="18">{escape(title or f"Gap family | N={c.num_agents}")}</text>',
            '<text x="24" y="47" font-family="sans-serif" font-size="12">Geometry preview only; controller success not evaluated</text>']
    for x0,y0,x1,y1 in instance.geometry.rectangles:
        x,y = xy((x0,y1))
        body.append(f'<rect x="{x}" y="{y}" width="{(x1-x0)*scale}" height="{(y1-y0)*scale}" fill="#666"/>')
    for wall in instance.geometry.walls[:4]:
        x,y=xy(wall[0]); a,b=xy(wall[1])
        body.append(f'<line x1="{x}" y1="{y}" x2="{a}" y2="{b}" stroke="black"/>')
    for i,(p,g) in enumerate(zip(instance.positions,instance.goals)):
        color = '#0072B2' if p[0] < c.barrier_x[0] else '#D55E00'
        x,y=xy(p); a,b=xy(g)
        body.append(f'<circle cx="{x}" cy="{y}" r="{c.agent_radius*scale}" fill="{color}"/>')
        body.append(f'<circle cx="{a}" cy="{b}" r="{c.agent_radius*scale}" fill="none" stroke="{color}" stroke-dasharray="2 2"/>')
    body.append(f'<text x="24" y="{height-15}" font-family="sans-serif" font-size="13">Solid: starts | Outline: goals | All {c.num_agents} robots must cross the barrier(s)</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'+''.join(body)+'</svg>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', type=int, default=10)
    parser.add_argument('--preset', choices=('gap','double','offset','wide','three_doors','clutter'), default='gap')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=Path, help='JSON Config; when given replaces preset/agents/seed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text())) if args.config else preset(args.preset,args.agents,args.seed)
    instance = build_instance(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'instance.json').write_text(json.dumps(instance.manifest(), indent=2)+'\n')
    (args.output/'preview.svg').write_text(svg(instance))
    print(json.dumps(instance.manifest()['validation'], indent=2))


if __name__ == '__main__':
    main()
