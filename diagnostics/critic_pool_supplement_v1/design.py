"""Prespecified independent training pool; no validation labels are read here."""
from common import *
import numpy as np
CONTROL_IDS=('centralized_sampling0','gradient_mpc0','DM0')
CONTROL_TYPES=('centralized_sampling','gradient_mpc','distributed_shooting')
CONTROL_NAMES=dict(zip(CONTROL_IDS,('集中式随机采样模型预测控制','梯度下降模型预测控制','分布式模型预测控制')))
ETA_KEYS=['safety']+[f'eta{i}'for i in range(16)]+[f'sE{i:02d}'for i in range(32)]
OLD=ROOT/'diagnostics/cross_scene_correction_v1';HET=ROOT/'diagnostics/hetero_scene_pipeline_v1'
FORBIDDEN=('lit_swap8_obstacles','lit_swap8_free','lit_circle_five','xsc_swap_n6','xsc_swap_n10','xsc_het_')
@lru_cache(None)
def menu():return copy.deepcopy(actions())
@lru_cache(None)
def controllers():
 d=read_evidence(OLD/'SELECTED_CONTROLLERS.json')['xsc_swap_n6'];out=[d[k]for k in CONTROL_TYPES];assert tuple(r['controller']for r in out)==CONTROL_IDS;return out

def specifications():
 rows=[]
 for n in [5,7,9,11]:rows.append(dict(name=f'xsc_tr_swap_n{n:02d}',group='A',N=n,r=.05,f=4.,v=.5))
 for n,k in [(6,3),(8,4),(10,5),(6,6),(8,8)]:rows.append(dict(name=f'xsc_tr_obst_n{n:02d}_k{k}',group='B',N=n,K=k,r=.05,f=4.,v=.5,a=.07,rho=.75))
 for n,r,f in [(5,.12,2.5),(5,.18,2.5),(7,.19,2.5),(8,.15,2.5),(8,.15,3.),(12,.11,2.5)]:rows.append(dict(name=f'xsc_tr_box_n{n:02d}_r{round(r*100):03d}_f{round(f*10):02d}',group='C',N=n,r=r,f=f,v=.5))
 for n,r,v in [(5,.25,.5),(5,.25,.2),(4,.3,.12)]:rows.append(dict(name=f'xsc_tr_bigbox_n{n:02d}_r{round(r*100):03d}_v{round(v*100):03d}',group='D',N=n,r=r,f=2.4,v=v))
 for w in [.16,.25,.32,.55]:rows.append(dict(name=f'xsc_tr_plus_w{round(w*100):03d}',group='E',N=4,r=.05,f=4.,v=.5,w=w))
 assert len(rows)==22;return rows
SCENE_NAMES={s['name']:(dict(A='无障碍对穿',B='静态障碍对穿',C='方形场地',D='大机器人方形场地',E='十字路口')[s['group']]+' '+s['name'])for s in specifications()}
NEW_NAMES=SCENE_NAMES;OLD_NAMES={}

def describe_eta(key,eta):
 title='仅安全过滤'if key=='safety'else'历史固定修正候选'if key.startswith('eta')else'扩展范围候选'if key.startswith('sE')else'历史菜单外候选'
 return title+'（'+key+'；'+', '.join(f'{v:.9g}'for v in eta)+'）'
