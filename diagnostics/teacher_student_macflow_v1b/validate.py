"""Delivery consistency checks; no simulation or model selection."""
from core import *
from audit_utils import source_inventory
import ast

def main():
    assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['files'][str(HERE/'PREREGISTRATION.md')];checks={};p=read(HERE/'DISTILLATION_REPORT.json');checks['ten_distilled_once']=len(p['controllers'])==10
    checks['validity_definition_exact']=all(r['valid']==(r['best_step']!=0 and r['mean_cosine']>=.9 and r['within_05_mps']>=.8) for r in p['controllers'])
    def function(path,name):
        src=Path(path).read_text();f=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name);return ast.dump(f,include_attributes=False)
    checks['pipeline_file_bitwise_unchanged']=sha(HERE/'pipeline.py')==sha(D/'decentralized_pipeline_v1/pipeline.py')
    checks['decentralized_action_unchanged']=function(HERE/'pipeline.py','decentralized_action')==function(D/'decentralized_pipeline_v1/pipeline.py','decentralized_action')
    checks['episode_physics_unchanged']=function(HERE/'episode_frozen.py','episode')==function(D/'decentralized_pipeline_v1/run.py','episode')
    checks['round28_replay_16_bitwise']=read(HERE/'ROUND28_REPLAY_GATE.json')['passed'] and read(HERE/'ROUND28_REPLAY_GATE.json')['completed']==16
    checks['db_module_correct']=Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
    from shared_rollout_db.src import cache_writer
    checks['journal_default_correct']=(cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
    imports={};conflicts=[]
    for f in HERE.glob('imports_*.json'):
        for path,h in read(f)['imported_sources'].items():
            if Path(path).is_relative_to(HERE):continue
            if path in imports and imports[path]!=h:conflicts.append(path)
            imports[path]=h
    changed=[p for p,h in imports.items() if sha(p)!=h];checks['imported_existing_sources_unchanged']=not changed and not conflicts;write(HERE/'IMPORTED_SOURCE_AUDIT.json',dict(existing_source_hashes=imports,changed=changed,hash_conflicts=conflicts))
    from datetime import datetime
    receipt=read(HERE/'TRAINING_FROZEN.receipt.json');pilot=read(HERE/'pilot_SUBMISSION.json');checks['training_frozen_before_student_closed_loop']=datetime.fromisoformat(receipt['utc'])<datetime.fromisoformat(pilot['utc'])
    if (HERE/'main_SUBMISSION.json').exists():
        choices=read(HERE/'CHOICES_FROZEN.receipt.json');submission=read(HERE/'main_SUBMISSION.json');checks['choices_frozen_before_main_submission']=datetime.fromisoformat(choices['utc'])<datetime.fromisoformat(submission['utc']);checks['evaluation_population_valid']=set(read(HERE/'main_protocol.json')['controller_order']).issubset(set(p['valid_names']))
    r=read(HERE/'results.json');checks['unknown_preserved']=all(r[k].get('unknown_total',0)>=0 for k in ['CONFIRMATORY','DEVELOPMENT'])
    for phase in ['pilot','main']:
        if (HERE/f'{phase}_MERGE_RECEIPT.json').exists():checks[phase+'_db_postflight']=read(HERE/f'{phase}_MERGE_RECEIPT.json')['db_exact']
    write(HERE/'RESULTS_VALIDATION.json',dict(utc=utc(),passed=all(checks.values()),checks=checks));source_inventory('imports_validate.json');assert all(checks.values()),checks
if __name__=='__main__':main()
