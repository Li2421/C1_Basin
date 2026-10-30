"""Load the literal historical numeric function bodies into private modules."""
from core import *
import ast,importlib.util

SNAP=ROOT/'diagnostics/hetero_set_critic_selfcheck_v1/workspace_snapshot_20261027'
def prepare():
    out=HERE/'accel_inputs';out.mkdir(exist_ok=True);rows=[]
    for source in [SNAP/'hetset/fastproj.py',SNAP/'hetset/accel.py',SNAP/'speed/np_threefry.py']:
        target=out/source.name;data=source.read_bytes()
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        rows.append(dict(source=str(source),target=str(target),sha256=sha(source)))
    dump(HERE/'ACCELERATION_INPUTS.json',dict(files=rows,loader_sha256=sha(__file__),load='Literal function ASTs from byte-identical archived modules; obsolete top-level imports/path injections excluded. Numeric statements unchanged; private runtime modules only.'))

def functions(path,requested,namespace):
    tree=ast.parse(Path(path).read_text());body=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in requested]
    assert len(body)==len(requested)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)

def enable(check=False):
    import swap_safety,swap_safety_user_fix_20261023 as SFIX,controllers_new as CN
    import shared_control.hard_projection as HP
    import clarabel
    from scipy import sparse
    spec=importlib.util.spec_from_file_location('private_np_threefry',HERE/'accel_inputs/np_threefry.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=dict(np=np,HP=HP,sparse=sparse,clarabel=clarabel,_P={})
    functions(HERE/'accel_inputs/fastproj.py',{'fast_project_velocity'},namespace)
    fn=namespace['fast_project_velocity']
    swap_safety.project_velocity=lambda *a,**kw:fn(*a,**kw,check=check)
    ns=dict(np=np,ORIG=dict(wp=CN.wall_penalties,lp=SFIX.original_problem))
    functions(HERE/'accel_inputs/accel.py',{'wall_penalties_pruned','local_problem_prescreen'},ns)
    CN.wall_penalties=ns['wall_penalties_pruned'];SFIX.original_problem=ns['local_problem_prescreen']
    RN.jax.random.fold_in=module.fold_in
    return module,fn

def accelerated_case(*args,**kwargs):
    from runner import case
    c=case(*args,**kwargs);c['controller_payload']['exact_acceleration']=dict(inputs=read(HERE/'ACCELERATION_INPUTS.json'),check=False)
    c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));return c
