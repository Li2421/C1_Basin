"""Gate-A/B pilot: 30 independent states plus explicit alternate orders."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .environment import Config, FourWayIntersectionEnv
from .expert import CentralizedExpert, CrossingOrder
from .scenario import sample_initial_state
from .rollout import crossing_order_signature
from .visualization import save_trajectory_svg

def run(count=30, split="development", output=None):
    cfg=Config(); expert=CentralizedExpert(); rows=[]; distinct=set(); chosen=set()
    for seed in range(count):
        p,v,meta=sample_initial_state(cfg,split,seed); env=FourWayIntersectionEnv(cfg);env.reset(p,v); plan=expert.plan(env)
        signature=crossing_order_signature(plan.positions); distinct.add(signature);chosen.add(plan.hypothesis.label)
        rows.append({"seed":seed,"success":plan.success,"collision":plan.collision,"steps":plan.episode_steps,"order":plan.hypothesis.label,"realized_order":signature,"min_pair":plan.min_inter_agent_surface_distance,"initial_draw":meta})
    # Same representative initial state deliberately admits disparate orders.
    p,v,_=sample_initial_state(cfg,split,0); modes=[]
    for o in ((0,2,1,3),(1,3,0,2),(0,1,2,3)):
        env=FourWayIntersectionEnv(cfg);env.reset(p,v); plan=expert.plan_hypothesis(env,CrossingOrder(o)); modes.append({"order":plan.hypothesis.label,"success":plan.success,"realized_order":crossing_order_signature(plan.positions)})
    result={"gate":"A/B","states":count,"expert_success_rate":sum(r["success"] for r in rows)/count,"collision_count":sum(r["collision"] for r in rows),"unique_selected_orders":sorted(chosen),"unique_realized_orders":sorted(distinct),"alternate_modes_same_state":modes,"rows":rows}
    if output:
        out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
        env=FourWayIntersectionEnv(cfg);env.reset(p,v);plan=expert.plan_hypothesis(env,CrossingOrder((0,2,1,3)));save_trajectory_svg(env,plan.positions,out.with_suffix('.svg'),terminal_reason=plan.terminal_reason,order_signature=crossing_order_signature(plan.positions))
    return result
if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--count",type=int,default=30);ap.add_argument("--output",default="diagnostics/four_way_intersection_stage1/expert_pilot.json");a=ap.parse_args();print(json.dumps(run(a.count,output=a.output),indent=2))
