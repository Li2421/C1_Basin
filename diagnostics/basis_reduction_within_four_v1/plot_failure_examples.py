"""Visualize saved simulation evidence only; never run a controller."""
import guard
from guard import HERE
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties


def main():
    read = lambda name: json.loads((HERE / name).read_text())
    scenes = {(e['type_id'], e['level']): e['scene'] for e in read('SCENES.json')['entries']}
    groups = {tuple(g['identity']): g for g in read('SEARCH_four_2_screen.json')['groups']}
    examples = [(7, 1, 'centralized_sampling', 2), (8, 1, 'centralized_sampling', 1), (16, 0, 'centralized_sampling', 5)]
    titles = ['障碍森林：8个障碍、10台机器人', '窄门对向交换：门宽0.3米、8台机器人', '货架仓库：巷宽0.4米、10台机器人']
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    records = []
    for ax, identity, title in zip(axes, examples, titles):
        group = groups[identity]
        assert not group['success'] and group['searched'] == 64
        query = group['queries'][-1]
        result = query['result']
        scene = scenes[identity[:2]]
        path = result['trace']
        with np.load(path) as trace:
            positions, goals = trace['positions'], trace['goals']
        radius = scene['config']['agent_radius']
        colors = plt.get_cmap('tab10')(np.arange(len(goals)) % 10)
        for wall in scene['walls']:
            a, b = np.asarray(wall)
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#263238', lw=2.2, zorder=3)
        for i, color in enumerate(colors):
            ax.plot(positions[:, i, 0], positions[:, i, 1], color=color, alpha=.4, lw=.8, zorder=2)
            ax.add_patch(Circle(positions[0, i], radius, fill=False, edgecolor=color, lw=1.3, ls='--', zorder=4))
            ax.add_patch(Circle(positions[-1, i], radius, color=color, alpha=.85, zorder=5))
            ax.scatter(*goals[i], marker='*', s=95, color=color, edgecolors='white', linewidths=.45, zorder=6)
            ax.text(*positions[-1, i], str(i + 1), ha='center', va='center', color='white', fontsize=6.7, zorder=7)
        x0, y0, x1, y1 = scene['bounds']
        ax.set(xlim=(x0 - .1, x1 + .1), ylim=(y0 - .1, y1 + .1), xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal')
        ax.set_title(title, fontproperties=font, fontsize=11)
        ax.grid(alpha=.12)
        records.append(dict(identity=list(identity), candidate=query['index'], rollout_uid=result['rollout_uid'],
                            trace=path, trace_sha256=hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest(),
                            scene_definition_sha256=scene['definition_sha256'], outcome=result['outcome']))
    fig.suptitle('四自由系数组合的阶段性未覆盖案例（G、R、T、C2；放行强度绑定|T|）\n虚线圆为起点，实心圆为最终位置，星号为真实目标', fontproperties=font, fontsize=12)
    fig.text(.5, .035, '仅展示各初态第64个候选的已保存轨迹；该组合在这些初态的64个候选均超时。没有重跑或更改仿真。',
             fontproperties=font, ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(HERE / 'failure_examples.png', dpi=180)
    fig.savefig(HERE / 'failure_examples.svg')
    (HERE / 'FAILURE_EXAMPLES.json').write_text(json.dumps(dict(examples=records, new_rollouts=0,
        selection='One observed failure per type; forest and door use first failed state, warehouse uses state5 where both frozen nominals remain uncovered by completed families.'), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
