"""Reproducible geometry/size audit, not a simulation batch or success benchmark."""
import json
from pathlib import Path

from .__main__ import preset, svg
from .scenario import build_instance


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    thumbnails = []
    for layout in ('gap','double','offset','wide','three_doors','clutter'):
        for n in (2,10,20,50,100):
            instance = build_instance(preset(layout,n))
            row = dict(preset=layout, num_agents=n, fingerprint=instance.config.fingerprint,
                       **instance.manifest()['validation'])
            rows.append(row)
            destination = args.output / f'{layout}_{n}'
            destination.mkdir(exist_ok=True)
            (destination/'instance.json').write_text(json.dumps(instance.manifest(),indent=2)+'\n')
            (destination/'preview.svg').write_text(svg(instance,f'{layout} | N={n}'))
            if layout == 'gap' or n == 100:
                thumbnails.append((layout,n,instance))
    # First row/column set: scale sweep; second set: obstacle variants at N=100.
    cell_w, cell_h = 620, 350
    body=[]
    for index,(layout,n,instance) in enumerate(thumbnails):
        panel=svg(instance,f'{layout} | N={n}',cell_w,cell_h)
        panel=panel.replace('<svg ',f'<svg x="{index%2*cell_w}" y="{index//2*cell_h}" ',1)
        body.append(panel)
    height=((len(body)+1)//2)*cell_h
    (args.output/'overview.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{cell_w*2}" height="{height}">'+''.join(body)+'</svg>\n')
    (args.output/'validation.json').write_text(json.dumps(dict(
        kind='geometry_audit_not_rollout_evaluation',instances=rows,
        checked_instances=len(rows),joint_feasibility='not_certified',
        controller_success='not_evaluated'),indent=2)+'\n')
    print(f'Checked {len(rows)} geometries; no policy evaluation. Overview: {args.output / "overview.svg"}')


if __name__ == '__main__':
    main()
