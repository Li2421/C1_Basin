"""Experiment-local, immutable operational snapshots; no rollout labels here."""
from common import *
import pickle

TARGET='roll_0feb972c720939d8680642faa2fc8e6610d37b2449794f2d1e777e024bdc1d16'


class CheckpointPause(BaseException):
    pass


class CheckpointStore:
    def __init__(self,c,pause=lambda:False,folder=None,forced_steps=()):
        self.case_hash=uid('checkpoint_case',c)
        self.folder=Path(folder)if folder is not None else HERE/'operational_checkpoints'/c['rollout_uid']
        assert self.folder.resolve().is_relative_to(HERE)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.sources={str(HERE/n):sha(HERE/n)for n in ('runtime_prospective.py','runtime_prospective_checkpointed.py','prospective_checkpoint_store.py')}
        self.pause=pause;self.forced_steps=set(forced_steps);self.last_saved=time.monotonic()
        self.resumed_from=None;self.saved=[]

    def load(self):
        records=[]
        for p in self.folder.glob('*.receipt.json'):
            row=read(p)
            if row['case_hash']!=self.case_hash or row['sources']!=self.sources:
                continue
            records.append((row['next_step'],p,row))
        if not records:return None
        _,receipt,row=max(records,key=lambda value:(value[0],str(value[1])))
        path=Path(row['path']);assert path.resolve().is_relative_to(self.folder.resolve())
        assert sha(path)==row['sha256'],('CHECKPOINT_HASH_MISMATCH',path)
        with path.open('rb')as f:state=pickle.load(f)
        assert state['next_step']==row['next_step']
        self.resumed_from=dict(receipt=str(receipt),receipt_sha256=sha(receipt),**row)
        return state

    def step(self,next_step,snapshot):
        paused=bool(self.pause());forced=next_step in self.forced_steps
        if not(paused or forced or time.monotonic()-self.last_saved>=60):return
        state=snapshot();state['next_step']=next_step
        path=self.folder/f'{next_step:06d}_{time.time_ns()}.pkl'
        tmp=path.with_suffix('.pkl.tmp')
        with tmp.open('wb')as f:
            pickle.dump(state,f,protocol=pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
        receipt=path.with_suffix('.receipt.json')
        write(receipt,dict(utc=utc(),path=str(path),sha256=sha(path),case_hash=self.case_hash,
            next_step=next_step,sources=self.sources,operational_only=True,scientific_label=None))
        self.saved.append(str(receipt));self.last_saved=time.monotonic()
        if paused or forced:raise CheckpointPause(str(receipt))

    def provenance(self):
        return dict(resumed_from=self.resumed_from,saved_checkpoint_receipts=self.saved,
            sources=self.sources,scientific_semantics='Canonical unchanged runtime loop; exact env/actor/history restored; original step-index RNG.',
            checkpoint_is_not_a_rollout_result=True)


def enabled(c):
    if c['rollout_uid']!=TARGET:return False
    gate=HERE/'CHECKPOINT_REPLAY_CHECK.json'
    if not gate.exists():return False
    proof=read(gate);assert proof['passed']
    for p,h in proof['implementation_sources'].items():assert sha(p)==h,p
    return True
