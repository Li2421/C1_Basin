"""Static resource-only variants; retain every sealed original file."""
from experiment import *
import ast
import re
from datetime import timedelta


def main():
    source={p.name:p.read_text() for suffix in ('*.py','*.sbatch') for p in HERE.glob(suffix)}
    modules={Path(n).stem:n for n in source if n.endswith('.py')}
    edges={}
    for name,text in source.items():
        deps=set(re.findall(r'\b([A-Za-z_]\w*\.(?:py|sbatch))\b',text))&source.keys()
        if name.endswith('.py'):
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module in modules:deps.add(modules[node.module])
                if isinstance(node,ast.Import):deps.update(modules[a.name] for a in node.names if a.name in modules)
        edges[name]=deps
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
    roots={v for v in entry.values() if isinstance(v,str) and v in source}
    roots.update(('resource_guard_corrections_release_cpu_only_20261023.py','cpu_release_guard_watchdog_20261023.py'))
    live=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            words=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if len(words)<3 or words[1]!=b'-B':continue
            path=Path(words[2].decode())
            if path.parent==HERE and path.name in source and proc.joinpath('stat').read_text().split(') ',1)[1].split()[0]!='Z':
                live.append(dict(pid=int(proc.name),script=path.name));roots.add(path.name)
        except (FileNotFoundError,PermissionError,UnicodeDecodeError):continue
    reachable=set(roots)
    while True:
        more=reachable|set().union(*(edges[n] for n in reachable))
        if more==reachable:break
        reachable=more
    seed='resources_corrections_release_20261023.py'
    changed={seed}
    while True:
        more=changed|{n for n in reachable if edges[n]&changed}
        if more==changed:break
        changed=more
    mapping={n:Path(n).stem+'_quota18_now_20261022'+Path(n).suffix for n in sorted(changed)}
    replacements=dict(mapping)
    replacements.update({Path(a).stem:Path(b).stem for a,b in mapping.items() if a.endswith('.py')})
    oldpolicy='RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json'
    newpolicy='RESOURCE_LIMIT_IMMEDIATE_18CPU_20261022.json'
    replacements[oldpolicy]=newpolicy
    def rewrite(text,m):
        pattern=re.compile(r'(?<!\w)('+ '|'.join(re.escape(k) for k in sorted(m,key=len,reverse=True))+r')(?!\w)')
        return pattern.sub(lambda match:m[match.group(0)],text)
    inverse={v:k for k,v in replacements.items()}
    files=[]
    for old,new in mapping.items():
        text=rewrite(source[old],replacements)
        if old==seed:
            assert 'CPU_LIMIT = 24' in text
            text=text.replace('CPU_LIMIT = 24','CPU_LIMIT = 18')
        restored=rewrite(text,inverse)
        if old==seed:restored=restored.replace('CPU_LIMIT = 18','CPU_LIMIT = 24')
        if old.endswith('.py'):assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(source[old])),old
        else:assert restored==source[old],old
        path=HERE/new
        if path.exists():assert path.read_text()==text,path
        else:path.write_text(text)
        files.append(path)
    policy=read(HERE/oldpolicy);now=utc()
    local_now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
    policy.update(utc=now,authority='Latest explicit user instruction: release six CPUs immediately; total18CPU/2GPU from now',release_at_local=local_now,release_at_utc=now,graceful_drain_at_local=local_now,graceful_drain_at_utc=now,deadline_fallback_at_utc=now,before=dict(total_cpu_shares=18,total_gpu_shards=2),after=dict(total_cpu_shares=18,total_gpu_shards=2),supersedes_readonly_policy=oldpolicy)
    write(HERE/newpolicy,policy)
    plan=dict(utc=utc(),mapping=mapping,live_local_processes=live,reachable_resource_modules=sorted(n for n in reachable if n.startswith('resources')),source_hashes={n:sha(HERE/n) for n in mapping},new_hashes={str(p):sha(p) for p in files},normalized_source_asts_equal=True,only_resource_policy_changed=True,new_policy=newpolicy,scientific_runtime_changes=False,new_rollouts=0)
    write(HERE/'CPU_LIMIT_IMMEDIATE_18CPU_SOURCE_MAP_20261022.json',plan)
    seal('CPU_LIMIT_IMMEDIATE_18CPU_CODE_20261022.receipt.json',[Path(__file__),HERE/newpolicy,HERE/'CPU_LIMIT_IMMEDIATE_18CPU_SOURCE_MAP_20261022.json',*files],normalized_source_asts_equal=True,only_resource_policy_changed=True,total_cpu_limit=18,gpu_limit=2)
    print(json.dumps(dict(created=len(files),resource_modules=plan['reachable_resource_modules'],mapping=mapping,live_processes=live),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
