# 训练状态的模型无关可选空间

全部为共同seed0样本内证据，先于任何真实critic拟合；不称为交叉拟合。源固定候选只按外层以外的源类型选择。

| 子集 | 权重 | 进度切换成功率% | 源多数票固定成功率% | 按状态样本内oracle% | 按控制器样本内oracle% |
| --- | --- | --- | --- | --- | --- |
| all | type_equal | 95.454 | 96.145 | 99.989 | 98.714 |
| all | controller_equal | 95.448 | 95.833 | 99.977 | 98.528 |
| learned | type_equal | 96.483 | 97.034 | 99.985 | 98.199 |
| learned | controller_equal | 96.354 | 96.507 | 99.969 | 98.009 |
| nonlearned | type_equal | 92.882 | 93.924 | 100.000 | 100.000 |
| nonlearned | controller_equal | 92.882 | 93.924 | 100.000 | 100.000 |

完整配对CI及相对两种固定基线的oracle空间见SELECTABLE_SPACE.json。每控制器最佳修法类与按类型计数分别见REPAIR_CLASS_BY_CONTROLLER.csv和REPAIR_CLASS_BY_TYPE.csv；平局按预注册进度锚点/菜单顺序，不掩盖并列最佳。
