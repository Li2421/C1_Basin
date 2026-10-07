"""Serial atomic merger, postflight and canonical-key validation."""
import argparse,json
from pathlib import Path
from . import seed_replication as replication
from . import permutation_outcomes as permutation
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import connect,transaction

def run(kind,replicate=0):
    if kind=='permutation':permutation.configure()
    if kind=='support':
        from .state_support import configure
        configure()
    if kind=='confirmation':
        from .support_confirmation import configure
        configure()
    if kind=='prefix':
        from .prefix_alias import configure
        configure()
    if kind=='fresh':
        from .fresh_controller_test import configure
        configure()
    if kind=='diversity':
        from .controller_diversity import configure
        configure()
    if kind=='diversity_confirmation':
        from .diversity_confirmation import configure
        configure()
    if kind=='function_support':
        from .function_support import configure
        configure()
    if kind=='function_confirmation':
        from .function_confirmation import configure,replica
        replica(replicate);configure()
    out=replication.OUT
    replication.merge()
    doc=preflight(out/'planned_rollouts.json');replication.write(out/'cache_postflight.json',doc)
    stats=replication.read(out/'alignment_audit.json')
    assert doc['summary']['genuinely_missing']==stats['numerical']
    assert not stats['conflict'] and not stats['missing']
    before=replication.read(out/'cache_preflight.json')['summary']
    excluded=replication.read(out/'preexisting_numerical_exclusions.json')['count']
    generated=before['genuinely_missing']-excluded
    with connect() as db,transaction(db):
        db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,reused_rollout_count=?,new_rollout_count=? WHERE experiment_uid=?',
            (before['exact_reusable']+before['partial_reusable'],generated,replication.EXP))
    replication.write(out/'working_state.json',dict(phase='postflight_complete',new_continuations=generated,
        journal_merged=True,postflight=doc['summary'],**stats))
    print(json.dumps({'new':generated,'postflight':doc['summary'],'alignment':stats},indent=2))
    if kind=='replication':
        from .replication_analysis import main
        main()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('replication','permutation','support','confirmation','prefix','fresh','diversity','diversity_confirmation','function_support','function_confirmation'));p.add_argument('--replicate',type=int,default=0);a=p.parse_args();run(a.kind,a.replicate)
