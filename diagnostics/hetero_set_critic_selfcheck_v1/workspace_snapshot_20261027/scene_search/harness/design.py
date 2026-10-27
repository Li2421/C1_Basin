"""Prespecified small test: no learned models or adaptive candidate search."""
from common import *
import numpy as np

CONTROL_IDS=('centralized_sampling0','gradient_mpc0','DM0')
CONTROL_NAMES={'centralized_sampling0':'集中式随机采样模型预测控制','gradient_mpc0':'梯度下降模型预测控制','DM0':'分布式模型预测控制'}
CONTROL_TYPES=('centralized_sampling','gradient_mpc','distributed_shooting')
ETA_KEYS=['safety']+[f'eta{i}'for i in range(16)]+['sE29','sE13','sE22','sE05','sE01','sE10','sE20','sE24']
NEW_NAMES={'xsc_div_size_r014':'大机器人','xsc_div_time_f20':'紧时限','xsc_div_center_s024':'中心障碍','xsc_div_offset_dx030':'偏心障碍'}
OLD_NAMES={'xsc_swap_n6':'对穿 6 机','xsc_swap_n4':'对穿 4 机','xsc_swap_n10':'对穿 10 机','xsc_swap_n12':'对穿 12 机','xsc_swap_n8_dense':'对穿 8 机加密',
 'lit_narrow_passage':'窄通道','lit_crowded_random':'拥挤随机','lit_circle_five':'圆形五机','lit_swap8_free':'八机无障碍','lit_swap8_obstacles':'八机静态障碍','lit_swap8_moving':'八机移动障碍'}
OLD=ROOT/'diagnostics/cross_scene_correction_v1'

@lru_cache(None)
def menu():
    source={a['key']:a for a in actions()};return[copy.deepcopy(source[k])for k in ETA_KEYS]

@lru_cache(None)
def controllers():
    selected=read_evidence(OLD/'SELECTED_CONTROLLERS.json')['xsc_swap_n6'];rows=[selected[t]for t in CONTROL_TYPES]
    assert tuple(r['controller']for r in rows)==CONTROL_IDS
    return rows

def old_tables():
    tables=read_evidence(OLD/'LABEL_TABLES.json');full_keys=[a['key']for a in actions()];cols=[full_keys.index(k)for k in ETA_KEYS];out={};audit={}
    for name in OLD_NAMES:
        row=tables[name];ci=[row['controllers'].index(c)for c in CONTROL_IDS];y=np.asarray(row['y'],float)[ci];u=np.asarray(row['unknown'],float)[ci]
        assert len(row['state_indices'])==(12 if name.startswith('xsc_')else 16)
        out[name]=dict(controllers=CONTROL_IDS,state_indices=row['state_indices'],eta_keys=ETA_KEYS,y=y[:,:,cols].tolist(),unknown=u[:,:,cols].tolist(),source='Historical exact frozen label table; no rollout.')
        q=y.mean((0,1));original=np.asarray(row['y']).mean((0,1));audit[name]=dict(full_original_controller_best=float(original.max()),selected25_original_controller_best=float(original[cols].max()),
            three_controller_full49_best=float(q.max()),three_controller_selected25_best=float(q[cols].max()),three_controller_full49_best_keys=[full_keys[i]for i in np.flatnonzero(q==q.max())])
    write(HERE/'OLD_LABEL_TABLES.json',out);write(HERE/'HISTORICAL_COHORT_CHECK.json',audit)
    write(HERE/'ETA_MENU.json',dict(keys=ETA_KEYS,candidates=menu(),controller_ids=CONTROL_IDS,controller_types=CONTROL_TYPES,no_model_training=True))
    return out

def describe_eta(key,eta):
    title='仅安全过滤'if key=='safety'else'历史固定修正候选'if key.startswith('eta')else'扩展范围候选'
    return title+'（'+key+'；'+', '.join(f'{v:.9g}'for v in eta)+'）'
