from core import *
from student import Student,inputs
from audit_utils import source_inventory
import ast,jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
def main():
    def nominal_ast(path):
        cls=next(x for x in ast.parse(Path(path).read_text()).body if isinstance(x,ast.ClassDef) and x.name=='Student');return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='nominal'),include_attributes=False)
    assert nominal_ast(HERE/'student.py')==nominal_ast(D/'teacher_student_macflow_v1b/student.py');rows=[]
    for i in [8,10,12,13]:
        s=Student(i);env=s.make_env(read(HERE/'EVAL_STATES.json')[0]);key=jax.random.PRNGKey(57);obs,actual,_=s.nominal(env,key);o=np.asarray(inputs(jnp.asarray(obs),s.mean,s.scale));out=[]
        for j in range(4):out.append(np.asarray(s.forward(o[j],jax.random.normal(jax.random.fold_in(key,j),(2,),dtype=jnp.float32))))
        raw=np.clip(np.asarray(out),-1.,1.);raw=raw*np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12));expected=s._base.to_world(np.asarray(raw,dtype=np.float64),env).copy();expected*=np.minimum(1.,env.config.max_speed/np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),1e-30));assert actual.tobytes()==expected.tobytes();rows.append(dict(index=i,old_local_formula_bitwise=True,checkpoint=str(s.checkpoint),sha256=sha(s.checkpoint)))
    write(HERE/'REUSE_RUNTIME_EQUIVALENCE.json',dict(status='PASS',nominal_ast_identical_to_round30b=True,local_students=rows,simulation_steps=0,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory('imports_reuse_equivalence.json')
if __name__=='__main__':main()
