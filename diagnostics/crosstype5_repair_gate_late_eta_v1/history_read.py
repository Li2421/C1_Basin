"""Fully parse requested history; preserve prior local-equivalence label provenance."""
from common import *
import csv

def main():
    assert (HERE/'INTEGRITY_START.json').exists();paths=read(HERE/'INPUT_MANIFEST.json')['paths'];records=[]
    for name in paths:
        p=Path(name)
        if not p.is_relative_to(D):continue
        assert not any(part.startswith(('decentralized_','teacher_student_','multiscene_')) for part in p.parts);r=dict(path=str(p),sha256=sha(p),size=p.stat().st_size)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:r['arrays']={k:dict(shape=z[k].shape,dtype=str(z[k].dtype),bytes=z[k].nbytes) for k in z.files}
        elif p.suffix=='.json':v=read(p);r['entries']=len(v) if isinstance(v,(list,dict)) else None
        elif p.suffix in ('.py','.md','.csv'):r['lines']=len(p.read_text().splitlines())
        records.append(r)
    c4=D/'crosstype4_no_repair_schedule_diagnosis_v1';extra=[c4/'EARLY_VALIDATION_POOL.json',c4/'B_evaluation_CASES.json',c4/'B_evaluation_protocol.json',c4/'EARLY_TERMINATION_EQUIVALENCE.receipt.json',c4/'A_ACTION_LINKS.json',c4/'A_CASES.json']
    additions=[]
    for p in extra:
        assert p.exists();value=read(p);additions.append(dict(path=str(p),sha256=sha(p),size=p.stat().st_size,entries=len(value),first_read_before_new_rollouts=True))
    write(HERE/'INPUT_ADDITIONS_01.json',dict(utc=utc(),files=additions,reason='Explicit historical raw-case identity and validated-source pools required for new reproduction and early-reuse gates; not unrelated parallel content'))
    write(HERE/'HISTORY_READ_INVENTORY.json',records+additions)
    write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),old34_labels_reused=True,critic_input_change_does_not_invalidate_execution_labels=True,old_local_equivalence_kept_outside_shared_DB=True,features_reused_without_probes=True,ten_missing_vectors_preserved=True,original_generator_box=[ [.5,-.5,0.],[1.25,.5,.75] ],generator_zero_support_policy='Out-of-box zero stages excluded from density fit as original; kept for critic/gate',old_cross_state_pair_index_CF_decomposition_not_used=True,new_gaps=['Exploration eta outside narrow frozen proposals:6 per controller-state','Requery semantics and4 labels per controller-state','Safety seed1 gaps only','Six-dimensional controller-conditioned schedule generator:old three-dimensional weights cannot answer','14-D critic and source-only repair gates','Matched9/8 coverage comparison','24 physically deduplicated fresh states; no closed loop before freeze'],input_based_parallel_audit=True,resource_limits=dict(cpu=24,gpu=5)))
    seal('HISTORY_AUDIT.receipt.json',[HERE/n for n in ('HISTORY_AUDIT.json','HISTORY_READ_INVENTORY.json','INPUT_MANIFEST.json','INPUT_ADDITIONS_01.json','history_read.py')],before_new_experiments=True)
    print('Fully parsed',len(records),'historical inputs;declared',len(additions),'raw identity/source pools',flush=True)
if __name__=='__main__':main()
