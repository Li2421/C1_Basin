"""Full-menu descriptive references from existing training labels only.

These estimates use the 32 training states and two label-generating instances
per type. They do not estimate an oracle on independent evaluation states.
No controller or correction is selected for deployment by this script.
"""
from experiment import *
from closure_scope import FLOW_NAMES
import numpy as np
import csv


def reference(labels):
    lower = np.maximum(labels, 0).astype(float)
    upper = np.where(labels < 0, 1., labels)
    half = np.arange(len(labels)) % 2
    value = 0.
    upper_value = 0.
    choices = []
    for h in (0, 1):
        fit = half == h
        test = ~fit
        choice = int(lower[fit].mean(0).argmax())
        choices.append(choice)
        value += lower[test, choice].sum()
        upper_value += upper[test, choice].sum()
    return dict(
        crossfit_lower=float(value / len(labels)),
        crossfit_same_choice_upper=float(upper_value / len(labels)),
        in_sample_best_lower=float(lower.mean(0).max()),
        in_sample_best_upper=float(upper.mean(0).max()),
        selected_menu_indices_by_fit_half=choices,
        unknown_labels=int((labels < 0).sum()),
        states=len(labels), actions=labels.shape[1])


def main():
    # Closed-form checks distinguish an overfit oracle from cross-fitting.
    toy = reference(np.asarray([[1, 0], [0, 1]]))
    assert toy['in_sample_best_lower'] == .5 and toy['crossfit_lower'] == 0.
    unknown = reference(np.asarray([[-1, 0], [-1, 0]]))
    assert unknown['crossfit_lower'] == 0. and unknown['crossfit_same_choice_upper'] == 1.
    rows = []
    gaps = {}
    inputs = []
    for scene in FLOW_NAMES:
        path = HERE / f'LEARNING_BUNDLE_{scene}.npz'
        if not path.exists():
            gap = HERE / f'LEARNING_GAP_{scene}.json'
            assert gap.exists(), scene
            gaps[scene] = read(gap)
            inputs.append(gap)
            continue
        inputs += [path, HERE / f'LEARNING_BUNDLE_{scene}.receipt.json']
        cap = ('capability_v2_' if scene == 'lit_narrow_passage' else 'capability_') + scene
        qpath = HERE / f'{cap}_QUALIFICATION.json'
        eligible = {q['controller'] for q in read(qpath) if q['qualified']}
        inputs.append(qpath)
        with np.load(path) as z:
            for labels, controller, typ in zip(z['labels'], z['controllers'], z['types']):
                rows.append(dict(scene=scene, controller=str(controller), type=str(typ),
                                 qualified=str(controller) in eligible, **reference(labels)))
    summary = []
    fields = ('crossfit_lower', 'crossfit_same_choice_upper',
              'in_sample_best_lower', 'in_sample_best_upper')
    for scene in FLOW_NAMES:
        rr = [r for r in rows if r['scene'] == scene and r['qualified']]
        for typ in [*sorted({r['type'] for r in rr}), 'type_equal']:
            selected = [r for r in rr if typ == 'type_equal' or r['type'] == typ]
            if not selected:
                continue
            types = sorted({r['type'] for r in selected})
            summary.append(dict(scene=scene, type=typ, controllers=len(selected),
                **{f:float(np.mean([np.mean([r[f] for r in selected if r['type'] == t])
                                     for t in types])) for f in fields}))
    result = dict(evidence='DEVELOPMENT', population='32 training states, seed0 only, two label-generating instances per type',
        menu='All twelve preregistered actions, including safety projection only',
        split='Sorted training states alternating halves; select on one half, evaluate on the other, then exchange; first menu index breaks ties',
        controller_rows=rows, qualified_type_equal_summary=summary, gaps=gaps,
        ci95=None, inference='Descriptive training-population references, not an independent held-out endpoint. Crossfit is not a guaranteed mathematical lower bound.',
        independent_evaluation_menu_regret=None,
        independent_evaluation_gap='Only frozen chosen actions were executed on evaluation states; complete-menu average and maximum regret are unobserved.',
        new_physical_steps=0)
    write(HERE / 'TRAINING_MENU_REFERENCE.json', result)
    with (HERE / 'TRAINING_MENU_REFERENCE.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    text = ['# 完整十二动作菜单的训练人群参照', '',
        '只重算已保存的32个训练状态、seed0标签，每类型两个标签生成实例。按排序状态奇偶分半，半数选动作、另半数评估再交换；同分选密封菜单中的首个动作。只描述训练人群，不作为独立评估成功率或部署选择，也不把交叉拟合值当严格数学下界。没有新增rollout。', '',
        '|场景|类型|合格实例数|交叉拟合成功率下界%|同选择的unknown上界%|样本内最佳下界%|样本内最佳上界%|',
        '|---|---|---|---|---|---|---|']
    names = {c['type_code']:c['type'] for c in read(HERE / 'CONTROLLER_MANIFEST.json')}
    names['type_equal'] = '类型等权合并'
    for r in summary:
        text.append('|' + '|'.join([FLOW_NAMES[r['scene']], names[r['type']],
            str(r['controllers']), *[f'{100*r[f]:.2f}' for f in fields]]) + '|')
    text += ['', '独立32个评估状态没有完整十二动作结果矩阵，完整菜单oracle及平均/最大遗憾仍不可识别。这里的训练标签不能填补这个缺口；未为事后oracle增加评估rollout。逐实例及未知上下界见CSV/JSON。']
    (HERE / 'TRAINING_MENU_REFERENCE.md').write_text('\n'.join(text) + '\n')
    seal('TRAINING_MENU_REFERENCE.receipt.json', [*inputs,
         HERE / 'TRAINING_MENU_REFERENCE.json', HERE / 'TRAINING_MENU_REFERENCE.csv',
         HERE / 'TRAINING_MENU_REFERENCE.md', HERE / 'training_menu_reference.py'],
         saved_labels_only=True, evaluation_labels_used_for_selection=False)


if __name__ == '__main__':
    main()
