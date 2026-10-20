"""Render frozen geometry manifests; no environment, controller or labels."""
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = dict(zip('ABCDEF', ['单门隔墙', '串联瓶颈走廊', '多门隔墙', '十字路口', '环岛', '房间与门']))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = HERE / 'geometry_figures'
    folder.mkdir(exist_ok=True)
    index = []
    for family, name in NAMES.items():
        source = HERE / ('PILOT_GEOMETRY_FROZEN.json' if family in 'BF' else f'FULL_{family}_GEOMETRY_FROZEN.json')
        states = json.loads(source.read_text())[family]
        assert len(states) == 72
        width, height = 1120, 70 + 18 * 158
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="20" y="28" font-size="21">{name}：冻结的训练与导航留出几何</text>',
               '<text x="20" y="52" font-size="13">灰色为实际障碍；圆点为清单中的四机器人起点，空心圆为目标；非执行轨迹图。</text>']
        colors = ['#0072b2', '#d55e00', '#009e73', '#cc79a7']
        for i, state in enumerate(states):
            ox, oy = 20 + i % 4 * 275, 76 + i // 4 * 158
            config = state['config']; lx, ly = config['half_length'], config['half_height']
            scale = min(250 / (2 * lx), 108 / (2 * ly))
            def x(v): return ox + 125 + v * scale
            def y(v): return oy + 54 - v * scale
            role = {'train': '复用训练', 'train_new': '新训练', 'navigation_holdout': '导航留出'}[state['role']]
            svg.append(f'<rect x="{x(-lx)}" y="{y(ly)}" width="{2*lx*scale}" height="{2*ly*scale}" fill="#fafafa" stroke="#222"/>')
            for a, b, c, d in state['rectangles']:
                svg.append(f'<rect x="{x(a)}" y="{y(d)}" width="{(c-a)*scale}" height="{(d-b)*scale}" fill="#666"/>')
            for k, (start, goal) in enumerate(zip(state['positions'], state['goals'])):
                for point, filled in [(start, True), (goal, False)]:
                    svg.append(f'<circle cx="{x(point[0])}" cy="{y(point[1])}" r="{config["agent_radius"]*scale}" fill="{colors[k] if filled else "white"}" stroke="{colors[k]}" stroke-width="1.2"/>')
            svg.append(f'<text x="{ox}" y="{oy+127}" font-size="12">{html.escape(role)}；几何种子 {state["geometry_seed"]}</text>')
        svg.append('</svg>')
        output = folder / f'{family}_frozen_geometries.svg'
        output.write_text('\n'.join(svg))
        index.append(dict(family=name, source=str(source), source_sha256=digest(source), output=str(output), output_sha256=digest(output), geometries=len(states), training=sum(s['role'] in ['train', 'train_new'] for s in states), navigation_holdout=sum(s['role']=='navigation_holdout' for s in states)))
    (HERE / 'GEOMETRY_FIGURES40_PROVENANCE.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), source_code_sha256=digest(Path(__file__)), artifacts=index, new_rollouts=0, physics_changes=0, controller_changes=0), ensure_ascii=False, indent=2)+'\n')
    (HERE / 'GEOMETRY_FIGURES40.md').write_text('冻结几何图册\n\n'+ '\n\n'.join(f'[{r["family"]}](geometry_figures/{Path(r["output"]).name})：{r["training"]} 个训练几何、{r["navigation_holdout"]} 个新导航留出几何。' for r in index)+'\n\n随机障碍场没有新训练清单；原训练与锁定评估几何重叠的缺口详见 LOCKED_G_TRAIN_GEOMETRY_GAP.receipt.json。\n')


if __name__ == '__main__':
    main()
