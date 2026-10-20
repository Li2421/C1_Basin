"""Join disjoint completed cohorts without simulation or copying trace contents."""
from common import *
import copy

def merge(target,phases):
    assert not (HERE/f'{target}_protocol.json').exists()
    combined=None;names=[];cases=[];seen=set();sources=[]
    for phase in phases:
        receipt=read(HERE/f'{phase}_COMPLETE.receipt.json')
        for path,digest in receipt['files'].items():assert sha(path)==digest
        p=read(HERE/f'{phase}_protocol.json');rows=read(HERE/f'{phase}_CASES.json')
        assert not set(names)&set(p['controller_order'])
        if combined is None:combined=copy.deepcopy(p)
        else:
            assert combined['states']==p['states']
            combined['profiles'].update(p['profiles']);combined['types'].update(p['types'])
        names.extend(p['controller_order']);cases.extend(rows)
        for c in rows:
            assert c['rollout_uid'] not in seen;seen.add(c['rollout_uid'])
            if p['phase']=='probe':
                source=HERE/phase/'raw'/f"{c['rollout_uid']}.json";assert source.exists()
                dest=HERE/target/'raw'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source)
        if p['phase']=='probe':
            for name in p['controller_order']:
                for ext in ('.npz','.receipt.json'):
                    source=HERE/phase/'initial_actions'/f'{name}{ext}';assert source.exists()
                    dest=HERE/target/'initial_actions'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source)
        sources.append(HERE/f'{phase}_COMPLETE.receipt.json')
    combined['controller_order']=names
    write(HERE/f'{target}_protocol.json',combined);write(HERE/f'{target}_CASES.json',cases)
    seal(f'{target}_MERGED.receipt.json',[HERE/f'{target}_protocol.json',HERE/f'{target}_CASES.json',*sources],
         no_execution=True,source_phases=phases)
    print('Merged',target,len(names),'controllers',len(cases),'requests',flush=True)

if __name__=='__main__':merge(sys.argv[1],sys.argv[2:])
