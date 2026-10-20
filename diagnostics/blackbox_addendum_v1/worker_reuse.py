from common import *
from runtime import execute as factor_execute
from controller_runtime import execute as controller_execute
from audit_utils import source_inventory
import fcntl,jax,traceback,time
jax.config.update('jax_enable_x64',False)
def main(phase,index):
    prereg();assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists();p=read(HERE/f'{phase}_protocol.json');rows=read(HERE/f'{phase}_MISSING_CASES.json');cache={};count=0;dest=HERE/phase;dest.mkdir(exist_ok=True)
    with (dest/'cursor').open('a+') as cursor:
        while True:
            fcntl.flock(cursor,fcntl.LOCK_EX);cursor.seek(0);s=cursor.read().strip();k=int(s) if s else 0
            if k>=len(rows):fcntl.flock(cursor,fcntl.LOCK_UN);break
            cursor.seek(0);cursor.truncate();cursor.write(str(k+1));cursor.flush();fcntl.flock(cursor,fcntl.LOCK_UN);c=rows[k];out=dest/'raw'/f"{c['rollout_uid']}.json";assert not out.exists(),'No retry of existing evidence';claim=dest/'claims'/f'{k}.json';assert not claim.exists();write(claim,dict(utc=utc(),worker=index,case=c))
            if c['engine']=='factor':r=factor_execute(c,p,cache,phase)
            elif c['engine']=='controller':r=controller_execute(c,p,cache,phase)
            elif c['engine'] in ['continuing_rhr','gap_mpc','gap_historical','timing']:
                import importlib
                module=importlib.import_module({'continuing_rhr':'rhr_reuse','gap_mpc':'gap_mpc','gap_historical':'gap_runtime','timing':'timing'}[c['engine']]);r=module.execute(c,p,cache,phase)
            else:raise ValueError(c['engine'])
            r['experiment_uid']=EXP;write(out,r);journal=cache_writer.append_journal([r],EXP,f'{phase}_{index}');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');count+=1
            if count%8==0:print(phase,index,count,flush=True)
    write(dest/f'worker_{index}.json',dict(utc=utc(),complete=True,new=count,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory(f'imports_{phase}_{index}.json')
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
