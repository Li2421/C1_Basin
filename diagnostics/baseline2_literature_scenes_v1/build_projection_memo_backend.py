from experiment import *
def main():
    old=(HERE/'reproduction_gate.py').read_text()
    text=old.replace('from projection_adapter import LiteratureSafetyFilter','from projection_memo import MemoSafetyFilter as LiteratureSafetyFilter').replace('REPRODUCTION_GATE','PROJECTION_MEMO_GATE')
    (HERE/'test_projection_memo_replay.py').write_text(text)
    oldrun=(HERE/'flow_runtime_memo.py').read_text();text=oldrun.replace('from projection_adapter import LiteratureSafetyFilter,constraints','from projection_memo import MemoSafetyFilter as LiteratureSafetyFilter').replace('A,l,kinds=constraints(snap,projector.config)','A,l,kinds=projector.linear_problem(snap)')
    assert text!=oldrun;(HERE/'flow_runtime_cached.py').write_text(text)
    worker=(HERE/'worker_flow_memo.py').read_text().replace('flow_runtime_memo','flow_runtime_cached').replace("proof=read(HERE/'MEMO_EQUIVALENCE.receipt.json')","proof=read(HERE/'PROJECTION_MEMO_EQUIVALENCE.receipt.json')")
    worker=worker.replace("proof='MEMO_EQUIVALENCE.receipt.json'","proof='PROJECTION_MEMO_EQUIVALENCE.receipt.json'")
    (HERE/'worker_flow_cached.py').write_text(worker)
    (HERE/'flow_cached.sbatch').write_text((HERE/'flow.sbatch').read_text().replace('worker_flow.py','worker_flow_cached.py'))
if __name__=='__main__':main()
