"""Final trajectory classifications, source integrity, and optional figure."""
from core import *
from collections import Counter
import datetime,subprocess,traceback

def goal_blocking(result,scene):
 from routes import guidance,route_geometry
 from admission import apply,AdmissionState
 try:
  with np.load(result['trace'])as tr:p=tr['positions'][-1];goals=tr['goals']
  env=RN.SwapEnv(scene['states'][result['state_index']],RN.LiteratureConfig(**scene['config']));geom=route_geometry(env.instance.geometry);guides,rem,paths=guidance(geom,p,goals,scene['route_guidance']);gates=read(HERE/'PASSAGES.json')['by_scene'][scene['name']]['passages']
  menu=dict(R_e=.8,R_c=.5,rho='fcfs',K=None,c1=0.,c2=0.,c3=0.)
  _,meta=apply(np.zeros_like(p),p,guides,rem,paths,env.config,gates,AdmissionState(len(gates)),menu)
  return dict(goal_blocked=any(any(x['goal_yield'])for x in meta),goal_blocking_details=[dict(passage=i,details=x['goal_yield_details'])for i,x in enumerate(meta)if any(x['goal_yield'])])
 except Exception as exc:return dict(goal_blocked=None,goal_blocking_error=repr(exc))

def failures(codes):
 from failure_geometry import failure_type
 from analysis_results import metrics,scenes,hard_keys
 records=[];scene_map=scenes();hard=hard_keys()
 for code in codes:
  for g in metrics(code)['groups']:
   if tuple(g['identity'])not in hard or g['success']or g['searched']!=64:continue
   for label,q in [('first_candidate',g['queries'][0]),('final_candidate',g['queries'][-1])]:
    r=q['result'];sc=scene_map[tuple(g['identity'][:2])];d=failure_type(r,sc);block=goal_blocking(r,sc)
    if block['goal_blocked']:d['inherited_category']=d['category'];d['category']='已到终点者挡路'
    records.append(dict(code=code,identity=g['identity'],candidate=q['index'],scope=label,**d,**block))
 dump(HERE/'FAILURE_ANALYSIS.json',dict(records=records,counts={code:{scope:dict(Counter(r['category']for r in records if r['code']==code and r['scope']==scope))for scope in ['first_candidate','final_candidate']}for code in codes},scope='Representative first/final executed candidates of cases with no success in64; no proof about every failed candidate or continuous parameter space.'))

def integrity():
 start=read(HERE/'INTEGRITY_START.json');changed={};imports={}
 for p in(HERE/'import_manifests').glob('*.json'):
  for path,expected in read(p).items():
   f=Path(path)
   if not f.is_relative_to(HERE):imports.setdefault(path,expected)
 for path,expected in imports.items():
  f=Path(path);actual=sha(f)if f.exists()else None
  if actual!=expected:changed[path]=dict(expected=expected,actual=actual)
 external=subprocess.check_output(['git','diff','--binary','--','.',':(exclude)diagnostics/basis_reduction_within_four_v1'])
 same=hashlib.sha256(external).hexdigest()==start['external_diff_sha256'];freeze=read(HERE/'DESIGN_FREEZE.json');design={n:dict(expected=h,actual=sha(HERE/n))for n,h in freeze['numerical_files'].items()if sha(HERE/n)!=h}
 generator_same=sha(HERE/'combinations.py')==freeze['combination_generator_sha256'];initial=freeze['initial_combinations']['combinations'];now={c['code']:c for c in read(HERE/'COMBINATIONS.json')['combinations']};initial_same=all(c==now[c['code']]for c in initial)
 passed=not changed and same and not design and generator_same and initial_same and not(HERE/'PROTECTION_VIOLATION.jsonl').exists()
 dump(HERE/'INTEGRITY_END.json',dict(passed=passed,changed_external_imports=changed,external_diff_unchanged=same,definition_changes=design,generator_unchanged=generator_same,initial_menus_unchanged=initial_same,external_import_hashes=imports,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 (HERE/'GIT_END_STATUS.txt').write_text(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True));(HERE/'GIT_END_DIFF.patch').write_bytes(subprocess.check_output(['git','diff','--binary']))
 return passed

def finish(out,gaps):
 minimum=out['minimum'];codes=['reference']+([minimum['code']]if minimum else[])
 try:failures(codes)
 except Exception as exc:gaps.append('失败分型未完成：'+repr(exc))
 if minimum and minimum['hard64']>=36:
  p=HERE/f'ALLSCENE_{minimum["code"]}.json'
  if not p.exists()or not read(p)['complete']:gaps.append('已触发的最小组合全场景验证未完成。')
 if not(HERE/'CORRECTIONS_ANALYSIS.json').exists():gaps.append('两处修正的旧合成一列对照尚未完成。')
 if(HERE/'FULL_SCENE_PLAN.json').exists():
  for i,code in enumerate(read(HERE/'FULL_SCENE_PLAN.json')['targets']):
   p=HERE/f'ALLSCENE_{code}.json'
   if not p.exists()or not read(p)['complete']:gaps.append(('非主线：预算条件下第二个全场景未完成：'if i else'已触发的全场景未完成：')+code)
 passed=integrity()
 if not passed:gaps.append('源码与冻结定义完整性检查未通过。')
 try:
  subprocess.run(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python',str(HERE/'plot_reduction.py')],check=True)
 except Exception as exc:gaps.append('非主线删减路径图未完成：'+repr(exc))
 from rebuild_candidates import rebuild
 rebuild()
 import importlib,analysis_results
 importlib.reload(analysis_results)
 analysis_results.write_report(final=True)
 with(HERE/'REPORT.md').open('a')as f:f.write('\n结束完整性：'+str(passed)+'。最小组合全场景触发条件为困难案例至少36/40。新运行数：'+str(read(HERE/'BUDGET.json')['actual_attempts'])+'，硬上限80000。\n')
 numeric_complete=not any(not g.startswith('非主线')for g in gaps)
 dump(HERE/'COMPLETE.json',dict(complete=numeric_complete and passed,required_numeric_work_complete=numeric_complete,integrity_passed=passed,finished=datetime.datetime.now().astimezone().isoformat(),actual_attempts=read(HERE/'BUDGET.json')['actual_attempts'],hard_limit=80000,minimum=minimum,report=str(HERE/'REPORT.md'),gaps=gaps))
 (HERE/'GAPS.md').write_text('# 证据缺口\n\n'+('\n'.join('- '+g for g in gaps)if gaps else'规定数值实验与交付已完成；有限候选未成功不是连续参数空间不可解证明。')+'\n')
 (HERE/'STATUS.md').write_text('# 当前状态\n\n'+('本轮完成。'if not gaps else'本轮结果已保存，存在证据缺口。')+'详见REPORT.md、REDUCTION.json、COMPLETE.json。\n')
