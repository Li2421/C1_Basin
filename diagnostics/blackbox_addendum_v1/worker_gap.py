from common import *
from audit_utils import source_inventory
import fcntl,jax,traceback,time
import new_benchmark_common,shared_control,bottleneck_family
for m in [new_benchmark_common,shared_control,bottleneck_family]:assert Path(m.__file__).resolve().is_relative_to(ROOT),m.__file__
jax.config.update('jax_enable_x64',False)
def main(phase,index):
    prereg();assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists();p=read(HERE/f'{phase}_protocol.json');rows=read(HERE/f'{phase}_MISSING_CASES.json');cache={};count=0;dest=HERE/phase;dest.mkdir(exist_ok=True)
    with (dest/'cursor').open('a+') as cursor:
        while True:
            fcntl.flock(cursor,fcntl.LOCK_EX);cursor.seek(0);s=cursor.read().strip();k=int(s) if s else 0
            if k>=len(rows):fcntl.flock(cursor,fcntl.LOCK_UN);break
            cursor.seek(0);cursor.truncate();cursor.write(str(k+1));cursor.flush();fcntl.flock(cursor,fcntl.LOCK_UN);c=rows[k];out=dest/'raw'/f"{c['rollout_uid']}.json";assert not out.exists(),'No retry of existing evidence';claim=dest/'claims'/f'{k}.json';assert not claim.exists();write(claim,dict(utc=utc(),worker=index,case=c))
            if c['engine'] in ['gap_mpc','gap_historical','timing']:
                import importlib
                module=importlib.import_module({'gap_mpc':'gap_mpc_v2','gap_historical':'gap_runtime','timing':'timing_v2'}[c['engine']]);r=module.execute(c,p,cache,phase)
            else:raise ValueError(c['engine'])
            r['experiment_uid']=EXP;write(out,r);journal=cache_writer.append_journal([r],EXP,f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');count+=1
            if count%8==0:print(phase,index,count,flush=True)
    write(dest/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory(f'imports_{phase}_{index}.json')
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
