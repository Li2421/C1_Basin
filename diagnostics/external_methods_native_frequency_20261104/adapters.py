"""Read-only source extraction keeps historical charted author controllers literal.

Importing the old core would install another experiment's write guard. Extract
only the unchanged controller declarations and ErrorContext instead. New temporal
semantics live entirely in clock_plant and worker, with separate identities.
"""
from native_io import *
import ast,types,copy,traceback
import numpy as np
from clock_plant import Config

def declarations(path,names,namespace):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return {n:namespace[n] for n in names}

def adapter(env,payload):
    method=payload['method']
    if method in ['circulation','mcbf']:
        paired=ROOT/'diagnostics/c1_v3_mcbf_circulation_paired_20261104';sys.path.insert(0,str(paired))
        from author_adapters import adapter as make
        return make(env,method),None
    root=Path(payload['legacy_root']);namespace=dict(np=np,sys=sys,types=types,copy=copy,traceback=traceback,HERE=root,
        RN=types.SimpleNamespace(LiteratureConfig=Config))
    declarations(root/'paper_methods/dac_author_si/adapter.py',['ErrorContext'],namespace)
    context=namespace['ErrorContext'](env.num_agents,unified=True)
    path=Path(payload['legacy_adapter'])
    declarations(path,['proxy','ChartedDAC','ChartedMGR'],namespace)
    cls=namespace['ChartedDAC' if method=='dac' else 'ChartedMGR']
    if payload.get('open_circle_extension'):
        namespace['charted']=types.SimpleNamespace(ChartedMGR=namespace['ChartedMGR'])
        declarations(root/'eps1_open_circle_paper_adapter.py',['OpenCircleMGR'],namespace)
        cls=namespace['OpenCircleMGR']
    return cls(env,payload['author_seed'],copy.deepcopy(payload['author_options']),context),context
