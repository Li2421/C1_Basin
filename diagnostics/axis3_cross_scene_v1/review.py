from common import *
import collections,ast,csv

def main():
 prereg();errors=[];counts={};unknowns={};ids=[]
 def check(ok,msg):
  if not ok:errors.append(msg)
 for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
 receipts=['ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json','R5_FROZEN.receipt.json','PROBE_INPUTS_FROZEN.receipt.json','TRAINING_INPUTS_FROZEN.receipt.json','TRAINING_IMPLEMENTATION_V2_FROZEN.receipt.json','PROBE_DURATION_AMENDMENT.receipt.json']+[sc+'_MODEL_FROZEN.receipt.json' for sc in SCENES]
 if (HERE/'DELIVERY_IMPLEMENTATION_FROZEN.receipt.json').exists():receipts.append('DELIVERY_IMPLEMENTATION_FROZEN.receipt.json')
 for name in receipts:
  for p,s in read(HERE/name)['files'].items():check(sha(p)==s,'Frozen hash mismatch '+p)
 with rollout_db.connect(True) as db:
  for phase in ['gate_v2','pilot','main']:
   rows=read(HERE/f'{phase}_CASES.json');counts[phase]=len(rows);unknowns[phase]=0;uids=[c['rollout_uid'] for c in rows];check(len(uids)==len(set(uids)),phase+' duplicate design');ids+=uids;pre=read(HERE/f'{phase}_PREFLIGHT.receipt.json');sub=read(HERE/f'{phase}_SUBMISSION.json');check(pre['utc']<sub['utc'],phase+' preflight chronology')
   for c in rows:
    path=HERE/phase/'raw'/f"{c['rollout_uid']}.json";check(path.exists(),'Missing raw '+str(path))
    if not path.exists():continue
    r=read(path);check(r['case']==c,'Raw/design mismatch '+str(path));check(not r['jax_enable_x64'],'New execution notFP32 '+str(path));check(sha(r['trace_path'])==r['trace_sha256'],'Trace digest mismatch '+str(path));unknowns[phase]+=r['numerical_failure'];check(r['deadlock']==int(r['outcome']=='deadlock'),'Deadlock taxonomy changed');check(r['timeout']==int(r['outcome']=='timeout'),'Timeout taxonomy changed');found=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();check(found is not None,'Missing authoritative UID '+c['rollout_uid'])
    if found:
     check(not found['conflict_quarantined'],'Quarantined '+c['rollout_uid'])
     for k in ['success','collision','timeout','deadlock','numerical_failure']:check(found[k]==r[k],'DB/raw outcome mismatch '+k)
   write(HERE/f'{phase}_POSTFLIGHT.json',planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json'))
  for phase in ['gate','gate_v2','pilot','probe','main']:
   for v in read(HERE/f'{phase}_REGISTRATION.receipt.json')['preexisting_row_hashes'].values():check(v['before']==v['after'],'Preexisting registry changed '+phase)
  probes=read(HERE/'probe_CASES.json');counts['probe_unlabeled']=len(probes)
  for c in probes:
   r=read(HERE/'probe/raw'/f"{c['rollout_uid']}.json");check(not r['terminal_labels_read'],'Probe terminal label read');check(not any(k in r for k in ['success','outcome','termination']),'Fake probe labels');check(sha(r['trace_path'])==r['trace_sha256'],'Probe trace hash');check(db.execute('SELECT COUNT(*) FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()[0]==0,'Fake probe rollout DB')
  summaries=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid=?',(EXP,)) if Path(r['path']).name.startswith('probe_')];check(len(summaries)==384 and all(r['classification']=='SUMMARY_ONLY' and r['rows_imported']==0 for r in summaries),'Probe SUMMARY_ONLY mismatch');dbcount=db.execute('SELECT COUNT(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];check(dbcount==7360,'Unexpected DB count')
 check(counts==dict(gate_v2=64,pilot=384,main=6912,probe_unlabeled=384),'Unexpected planned counts');check(len(ids)==len(set(ids)),'Cross-stage duplicate identity');counter=collections.Counter()
 for path in (cache_writer.ROOT/'journals'/EXP).iterdir():
  if path.suffix not in ['.merged','.jsonl']:continue
  for line in path.read_text().splitlines():counter[json.loads(line)['case']['rollout_uid']]+=1
 expected=set(ids)|{c['rollout_uid'] for c in probes};check(set(counter)==expected and all(n==1 for n in counter.values()),'Repeated/missing journal IDs');write(HERE/'JOURNAL_UNIQUENESS_AUDIT.json',dict(utc=utc(),records=sum(counter.values()),unique=len(counter),expected=len(expected),duplicates={k:v for k,v in counter.items() if v!=1},passed=set(counter)==expected and all(n==1 for n in counter.values())))
 gate=read(HERE/'CONSISTENCY_GATE.json')['scenes'];check(not gate['toy_give_way']['passed'] and gate['toy_give_way']['passed_count']==0,'Toy precision gap hidden');check(all(gate[s]['passed_count']==16 and gate[s]['passed'] for s in SCENES[1:]),'Scene source gate failed');check(len(read(HERE/'SCENE_INDEX_REPAIR.json')['valid_reused_rollouts'])==8,'Valid gate reuse count')
 for before,after in [('PROBE_DURATION_AMENDMENT.receipt.json','probe_SUBMISSION.json'),('R5_FROZEN.receipt.json','TRAINING_SUBMISSIONS_V2.json'),('CHOICES_FROZEN.receipt.json','main_SUBMISSION.json'),('ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','main_SUBMISSION.json')]:check(read(HERE/before)['utc']<read(HERE/after)['utc'],'Freeze chronology '+before)
 source=np.load(HERE/'SOURCE_TRAINING_DATA.npz');check(np.all((source['y']>=0)&(source['y']<=source['n'])&(source['n']<=source['q'])),'Source counts invalid');isolation={}
 for target in SCENES:
  selected=read(HERE/'fits'/target/'SELECTED_STEP.json');check(target not in selected['source_scenes'] and len(selected['source_scenes'])==3,'LOSO source scene leak');check(selected['target_rows_in_train']==0 and selected['target_rows_in_cv']==0,'LOSO target labels used');loss=[]
  for held in selected['source_scenes']:
   for seed in [17,23,41]:
    fit=read(HERE/'fits'/target/'cv'/held/f'seed{seed}'/'fit.json');check(fit['train_rows']==int(((source['scenes']!=target)&(source['scenes']!=held)).sum()),'CV train row membership');check(fit['held_rows']==int((source['scenes']==held).sum()),'CV held membership');loss.append(fit['validation_nll'])
  means=np.mean(loss,axis=0);grid=read(HERE/'DESIGN_CONFIG.json')['steps'];check(selected['selected_step']==grid[int(means.argmin())],'Selected step rule mismatch');isolation[target]=dict(source_scenes=selected['source_scenes'],selected_step=selected['selected_step'],target_rows_in_train=0,target_rows_in_cv=0)
 write(HERE/'LOSO_ISOLATION_AUDIT.json',dict(utc=utc(),passed=True if not any('LOSO' in x or 'CV ' in x for x in errors) else False,targets=isolation))
 results=read(HERE/'results.json');frozen=read(HERE/'FROZEN_CHOICES.json')
 for scene in SCENES[1:]:
  z=np.load(HERE/f'{scene}_LABELS.npz');names=z['controller_names'].tolist();choices=np.asarray([ARMS.index(frozen['controllers'][scene][n]) for n in names]);value=z['lower'][np.arange(len(names))[:,None],np.arange(32)[None,:],choices[:,None]].mean()*100;check(abs(value-results[scene]['methods']['LOSO critic']['success_pct'][0])<1e-10,'Independent LOSO success mismatch '+scene)
 check(results['toy_give_way']['status']=='GAP' and results['pooled']['XS1']['status']=='INCONCLUSIVE','Missing Toy four-target claim');few=read(HERE/'FEWSHOT_RESULTS.json')
 for scene in SCENES[1:]:check(not (set(few[scene]['calibration_original_uids'])&set(few[scene]['evaluation_original_uids'])),'Fewshot calibration/evaluation leakage')
 rows=list(csv.DictReader((HERE/'AXIS_TABLE.csv').open()));check(len(rows)==5,'Required four scenes+pooled table missing');required=['REPORT.md','HANDOFF.md','AXIS_TABLE.md','AXIS_TABLE.csv','METHOD_METRICS.csv','GAPS.json','FEWSHOT_RESULTS.json','FEWSHOT_CURVES.csv','fewshot_curves.svg','fewshot_curves.png','fewshot_curves.pdf','INTEGRITY_DETAILS.json','SOURCE_LABEL_PROVENANCE.json','LOSO_ISOLATION_AUDIT.json','JOURNAL_UNIQUENESS_AUDIT.json']
 for p in required:check((HERE/p).exists(),'Missing delivery '+p)
 comp=read(HERE/'INTEGRITY_COMPARISON.json');check(('未完全通过' in (HERE/'REPORT.md').read_text()) if not comp['passed'] else ('审计通过' in (HERE/'REPORT.md').read_text()),'Integrity disclosure mismatch');review=dict(utc=utc(),passed=not errors,errors=errors,counts=counts,unknowns=unknowns,authoritative_DB_rollouts=dbcount,probe_summary_only=len(summaries),all_trace_digests_checked=True,source_integrity_passed=comp['passed'],Toy_gap_preserved=True,unknown_retries=0,source_isolation_verified=True);write(HERE/'INDEPENDENT_DELIVERY_REVIEW.json',review);assert not errors,errors;seal('NUMERICAL_COMPLETE.receipt.json',[HERE/p for p in required]+[HERE/'results.json',HERE/'INDEPENDENT_DELIVERY_REVIEW.json',HERE/'INTEGRITY_COMPARISON.json'],numerical_complete=True,source_integrity_passed=comp['passed'],Toy_prospective_gap=True,counts=counts,unknowns=unknowns);print('Round34 numerical review PASSED',counts,unknowns,flush=True)
if __name__=='__main__':main()
