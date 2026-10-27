"""Prespecified eight scenes, exact49 menu and original fast-controller profiles."""
from common import *
import numpy as np
CONTROL_IDS=('centralized_sampling0','gradient_mpc0','DM0')
CONTROL_TYPES=('centralized_sampling','gradient_mpc','distributed_shooting')
CONTROL_NAMES=dict(zip(CONTROL_IDS,('集中式随机采样模型预测控制','梯度下降模型预测控制','分布式模型预测控制')))
ETA_KEYS=['safety']+[f'eta{i}'for i in range(16)]+[f'sE{i:02d}'for i in range(32)]
OLD=ROOT/'diagnostics/cross_scene_correction_v1'
TEST1=ROOT/'diagnostics/eta_diverse_scene_test_v1'
SCENE_NAMES={'lit_swap8_obstacles':'八机静态障碍','xsc_swap_n10':'对穿10机','lit_swap8_free':'八机无障碍','xsc_swap_n6':'对穿6机','xsc_het_plus_w020':'十字路口半宽0.20','xsc_het_plus_w040':'十字路口半宽0.40','lit_circle_five':'圆形五机','xsc_het_box_n10_r013_f25':'方形场地10机'}
NEW_NAMES={n:t for n,t in SCENE_NAMES.items()if n.startswith('xsc_het_')}
OLD_NAMES={n:t for n,t in SCENE_NAMES.items()if n not in NEW_NAMES}
PAIRED={a:b for a,b in [('lit_swap8_obstacles','xsc_swap_n10'),('xsc_swap_n10','lit_swap8_obstacles'),('lit_swap8_free','xsc_swap_n6'),('xsc_swap_n6','lit_swap8_free'),('xsc_het_plus_w020','xsc_het_plus_w040'),('xsc_het_plus_w040','xsc_het_plus_w020'),('lit_circle_five','xsc_het_box_n10_r013_f25'),('xsc_het_box_n10_r013_f25','lit_circle_five')]}
@lru_cache(None)
def menu():return copy.deepcopy(actions())
@lru_cache(None)
def controllers():
 s=read_evidence(OLD/'SELECTED_CONTROLLERS.json')['xsc_swap_n6'];out=[s[t]for t in CONTROL_TYPES];assert tuple(c['controller']for c in out)==CONTROL_IDS;return out

def old_tables():
 tables=read_evidence(OLD/'LABEL_TABLES.json');out={}
 for n in OLD_NAMES:
  row=tables[n];ci=[row['controllers'].index(c)for c in CONTROL_IDS]
  out[n]=dict(controllers=list(CONTROL_IDS),state_indices=row['state_indices'],eta_keys=ETA_KEYS,y=np.asarray(row['y'])[ci].tolist(),unknown=np.asarray(row['unknown'])[ci].tolist(),source=str(OLD/'LABEL_TABLES.json'))
 write(HERE/'OLD_LABEL_TABLES.json',out);write(HERE/'ETA_MENU.json',dict(keys=ETA_KEYS,candidates=menu(),controller_ids=CONTROL_IDS,controller_types=CONTROL_TYPES));return out

def describe_eta(key,eta):
 title='仅安全过滤'if key=='safety'else'历史固定修正候选'if key.startswith('eta')else'扩展范围候选'if key.startswith('sE')else'生成器提案'
 return title+'（'+key+'；'+', '.join(f'{v:.9g}'for v in eta)+'）'
