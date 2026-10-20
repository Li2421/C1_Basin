"""Synthetic-only inference checks before any corrective target labels."""
from common import *
from corrective_analysis40 import cluster_bootstrap,crossfit_oracle
from fewshot_choices40 import beta_gt
from selector_train40 import Critic
import jax,jax.numpy as jnp
import ast

def main():
    jax.config.update('jax_enable_x64',False);Y=np.zeros((3,32,9,2),np.float32);Y[:,:,3]=1
    methods={'winner':np.eye(9)[[3,3,3]],'loser':np.eye(9)[[2,2,2]]};boot=cluster_bootstrap(Y,methods,202610174040,repeats=128)
    assert np.array_equal(boot['winner'],np.ones(128)) and np.array_equal(boot['loser'],np.zeros(128))
    assert np.array_equal(crossfit_oracle(Y,list(range(2,9))),np.ones((3,32)))
    Y[:]=.5;boot=cluster_bootstrap(Y,methods,202610174041,repeats=128);assert np.array_equal(boot['winner']-boot['loser'],np.zeros(128))
    assert abs(beta_gt(0,1,0,1)-.5)<1e-12 and beta_gt(16,16,0,16)>.999
    net=Critic();params=net.init(jax.random.PRNGKey(17),jnp.zeros((1,33),jnp.float32));assert net.apply(params,jnp.zeros((2,33))).shape==(2,)
    def get(path,name):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    old=D/'axis3_cross_scene_v1/train_v2.py';new=HERE/'selector_train40.py';assert ast.dump(get(old,'Critic'),include_attributes=False)==ast.dump(get(new,'Critic'),include_attributes=False)
    text=new.read_text().replace('from selector_features40 import fit_transform,apply_transform','from features import fit_transform,apply_transform').replace("HERE/'SELECTOR40_CONFIG_FROZEN.json'","HERE/'DESIGN_CONFIG.json'").replace('jnp.zeros((1,33),jnp.float32)','jnp.zeros((1,24),jnp.float32)').replace("HERE/'incomplete_selector_fits40'","HERE/'incomplete_fits_before_gres_repair'").replace("HERE/'fits40'","HERE/'fits'")
    restored=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='fit');assert ast.dump(restored,include_attributes=False)==ast.dump(get(old,'fit'),include_attributes=False)
    write(HERE/'SELECTOR_INFERENCE40_UNIT_TEST.json',dict(utc=utc(),passed=True,tests=9,synthetic_only=True,physical_steps=0,real_task_labels_read=0,fit_architecture_optimizer_validation_rule_AST_identical_except_paths_and_input_dimension=True,jax_enable_x64=False,source_sha256={n:sha(HERE/n) for n in ['corrective_analysis40.py','fewshot_choices40.py','selector_train40.py','selector_features40.py']}))

if __name__=='__main__':main()
