from experiment import *
from registry import save
from paper_secondary import run

def main(stage):
    prereg();cases=read(HERE/f'{stage}_MISSING_CASES.json');claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True)
    for c in cases:
        if (HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists():continue
        claim=claims/c['rollout_uid']
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),rollout_uids=[c['rollout_uid']]))
        assert all(sha(HERE/n)==h for n,h in c['controller_payload']['source_hashes'].items())
        result,arrays=run(c);save(stage,c,result,arrays)
        write(claim/'COMPLETE.json',dict(utc=utc()))
        print(c['state']['scene'],c['state']['index'],c['method'],result['outcome'],result['elapsed_seconds'],flush=True)

if __name__=='__main__':main(sys.argv[1])
