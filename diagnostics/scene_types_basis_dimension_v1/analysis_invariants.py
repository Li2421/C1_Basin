"""Synthetic inference boundary checks; no simulator/imported numerical backend."""
import guard
from guard import HERE
import ast,hashlib,json

source=HERE/'discovery.py'
tree=ast.parse(source.read_text())
selected={'support','mask','names','key','successful_supports','summarize_supports'}
module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in selected],type_ignores=[])
new=('T1','T2','T3','T4','T5','P2','R2','R3','Y1','Y2','Y3','Y4','S1','C1','C2')
namespace=dict(ALL=('G','P','R')+new,NEW=new)
exec(compile(module,str(source),'exec'),namespace)
cohort=[(dict(type_id=t),dict(type_code='synthetic'),0)for t in range(1,5)]
def row(t,eta,coefficients):return dict(type_id=t,controller='synthetic',state_index=0,success=True,spec=dict(eta=eta,coefficients=coefficients))
rows=[row(1,[.5,0.,0.],{}),row(2,[0.,0.,0.],{'T1':.75}),row(4,[0.,0.,0.],{})]
summary,dp=namespace['summarize_supports'](rows,cohort)
assert not summary['box_compatible_global_cover_found']
assert summary['global_dimension']is None
assert summary['per_type']['3']['dimension']is None
assert summary['per_type']['3']['minimal_supports']==[]
assert summary['per_type']['4']['dimension']==0
assert summary['per_type']['4']['minimal_supports']==[[]]
assert not dp[-1]&(1<<1),'G=0 success must not enter a G-present family'
assert namespace['support'](dict(eta=[.5,0.,0.],coefficients={'T1':0.}))==('G',)
result=dict(passed=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checks=['G-positive and G-absent boxes are separate','uncovered type has no inferred zero-dimensional success','genuine no-correction success is a valid zero-dimensional witness','declared zero coefficients do not increase effective dimension'],synthetic_summary=summary,no_rollouts=True)
(HERE/'ANALYSIS_INVARIANTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('analysis inference boundary checks PASS',flush=True)
