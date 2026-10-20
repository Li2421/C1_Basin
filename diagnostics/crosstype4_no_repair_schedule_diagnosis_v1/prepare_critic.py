"""Copy crosstype3 training implementation with action-dimension/cohort changes."""
from common import *
import difflib

def main():
    src=D/'crosstype3_unseen_controller_schedule_v1';sources=[]
    for name in ('legacy_critic_model.py','critic_model.py','train_critic.py','select_models.py','inference.py'):
        target=HERE/name;assert not target.exists();old=(src/name).read_text();text=old
        if name=='critic_model.py':
            text=text.replace('jnp.zeros((1,10))','jnp.zeros((1,13))').replace('diag=index//40==index//5%8','diag=index//54==index//6%9').replace('pair=index//5','pair=index//6').replace('index[...,None]%5!=index[...,None,:]%5','index[...,None]%6!=index[...,None,:]%6')
            text=text.replace("rank={}\n", "zero=(index//54==8)|(index//6%9==8)\n    groups['zero_vs_nonzero']=zero[...,None]!=zero[...,None,:]\n    rank={}\n")
        if name=='train_critic.py':
            text=text.replace('(104,48,26,1)','(122,48,34,1)').replace('(104,48,26,10)','(122,48,34,13)').replace('(104,24)','(122,24)').replace('==16 and','==19 and').replace('reshape(-1,10)','reshape(-1,13)')
            text=text.replace("    assert np.all(data['candidate_indices'][:,:,:8]//40==data['candidate_indices'][:,:,:8]//5%8)\n    assert np.all(data['candidate_indices'][:,:,8:]//40!=data['candidate_indices'][:,:,8:]//5%8)","    idx=data['candidate_indices'];assert np.all(idx[:,:,:8]//54==idx[:,:,:8]//6%9)\n    assert np.all(idx[:,:,26]==483)")
            text=text.replace("assert read(HERE/'CRITIC_TESTS.json')['passed'] and (HERE/'SELECTABLE_SPACE.receipt.json').exists()", "assert read(HERE/'CRITIC_TESTS.json')['passed'] and (HERE/'CANDIDATES.receipt.json').exists()")
            start=text.index('    from families import family_groups');end=text.index('    for source in sorted(sources',start)
            text=text[:start]+"    sources=set()\n    for count in (17,18,19):sources.update(itertools.combinations(types,count))\n    assert len(sources)==191\n"+text[end:]
            text=text.replace('len(tasks)==312','len(tasks)==382').replace('source_sets=156','source_sets=191').replace("HERE/'SELECTABLE_SPACE.receipt.json'","HERE/'CANDIDATES.receipt.json'")
        if name=='select_models.py':
            text=text.replace('(104,48)','(122,48)').replace('(3,104,48,26)','(3,122,48,34)').replace('diagonal=indices//40==indices//5%8;fixed=diagonal|(indices%5==3)','diagonal=indices//54==indices//6%9;fixed=diagonal|(indices%6==3)')
            text=text.replace("return dict(self_schedule=pick(score,np.ones_like(diagonal)),fixed500=pick(score,fixed),no_schedule=pick(score,diagonal),conservative=pick(score-sd,np.ones_like(diagonal)))", "zero=(indices//54==8)|(indices//6%9==8);progress=indices%6==5\n    return dict(self_schedule=pick(score,np.ones_like(diagonal)),fixed500=pick(score,fixed),no_schedule=pick(score,diagonal),conservative=pick(score-sd,np.ones_like(diagonal)),no_progress=pick(score,~progress),no_zero=pick(score,~zero),legacy_fixed=pick(score,~zero&~progress))")
            text=text.replace("'conservative','no_controller','single_branch'", "'conservative','no_progress','no_zero','legacy_fixed','no_controller','single_branch'")
        if name=='inference.py':text=text.replace('reshape(-1,10)','reshape(-1,13)')
        target.write_text(text);sources.append(dict(source=str(src/name),source_sha256=sha(src/name),copy=str(target),copy_sha256=sha(target),diff=''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True)))))
    write(HERE/'B_CRITIC_COPY_SOURCES.json',sources)
    print('Copied source-only critic, 191 source sets/382 fits; action13; no new architecture',flush=True)
if __name__=='__main__':main()
