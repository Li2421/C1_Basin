#!/usr/bin/env python3
"""Frozen untouched-test evaluation for the joint OrthoFlow3 generator/critic.

The script has four deliberately separated phases:

``freeze`` creates every generator proposal, critic score/choice and the
validation-only deformation threshold before any test outcome is queried;
``preflight`` queries the shared rollout database; ``run`` executes only
missing exact outcomes in deterministic shards; and ``finalize`` performs
matched outcome/deformation analysis without changing a controller artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
LEARN = ROOT / "diagnostics/orthoflow3_generator_critic_v1"
DB_PATH = ROOT / "shared_rollout_db/rollout.sqlite"

from diagnostics.orthoflow3_generator_critic_v1.train_evaluate import (
    CENTER, HIGH, K, LOW, RADIUS, Critic, Generator, METHOD,
    eta_from_noise, eta_mean, merge_initialized, scalar_environment_fields,
    stable_int,
)
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import (
    CertifiedHardSafetyFilter,
)
from diagnostics.orthoflow3_db_shared_mode_transfer_v1 import run_db
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as db_old
from new_benchmark_common.basin_dataset_v1 import CONDITIONING_FLOW_ROOT, _environment_descriptor
from new_benchmark_common.safety_eta3 import (
    FROZEN_EVAL_SEEDS, FUTURE_ROOT, ScenarioRuntime, canonical, cached_rows,
    eta_identity, normalized_eta, sha256_file, state_token, uid,
)
from shared_rollout_db.src.rollout_db import connect, initialize, lookup_exact

SCENARIOS = ("double_bottleneck", "four_way_intersection", "ring_exchange")
SEEDS = tuple(range(16))
GENERATOR_SEED = 41
TAU_CRITIC = 15 / 16
DOUBLE_FUTURE_ROOT = 2026100117
EXPERIMENT = "orthoflow3_generator_critic_frozen_test_v1"
GENERATOR_CKPT = LEARN / "generator/seed41/checkpoint.msgpack"
CRITIC_CKPT = LEARN / "critic/seed23/checkpoint.msgpack"
NORMALIZATION = LEARN / "normalization.json"
DOUBLE_POOLS = {
    "existing_untouched_test": ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
    "fresh_untouched_test": ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool",
}
DOUBLE_HARD = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
FIXED_ETA = {
    "double_bottleneck": [0.5146302983186439, 0.4425638246215209, 0.0011432715489280154],
    "four_way_intersection": [0.7803556208964437, -0.24812070094048977, 0.02150735165923834],
    "ring_exchange": [1.0885793089400977, -0.06548301875591278, 0.1545611140318215],
}


def dump(path: Path | str, value: Any) -> None:
    path = OUT / path if isinstance(path, str) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path: Path | str) -> Any:
    return json.loads((OUT / path if isinstance(path, str) else path).read_text())


def file_hash(path: Path) -> str:
    return sha256_file(path)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def stats(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    if not len(x):
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": int(len(x)), "mean": float(np.mean(x)),
            "median": float(np.median(x)), "p95": float(np.quantile(x, .95))}


def outcome_class(row: dict[str, Any]) -> str:
    if row.get("numerical_failure"): return "numerical_failure"
    if row.get("success"): return "success"
    if row.get("collision"): return "collision"
    if row.get("deadlock"): return "deadlock"
    return "timeout" if row.get("timeout") else str(row.get("outcome", "other"))


def register_extra_controller(runtime: Any, name: str, rng: dict[str, Any]) -> None:
    base = runtime.controllers["orthoflow3"]["payload"]
    payload = {**base, "chain": name, "rng": rng,
               "conditioning": "frozen_test_state_generator_proposal_v1"}
    controller = {"uid": uid("ctl", payload), "payload": payload}
    runtime.controllers[name] = controller
    with connect() as con:
        con.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
          flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
          success_semantics_version,conditioning_version,rng_semantics_version,
          config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
          (controller["uid"], runtime.scenario_uid, runtime.checkpoint_sha,
           runtime.orthoflow_hash if "orthoflow3" in name else None,
           runtime.safety_hash, str(runtime.config.max_steps), str(runtime.config.dt),
           base["success_semantics"], payload["conditioning"], canonical(rng),
           canonical(payload), "EXACT_PROFILE"))
        con.commit()


class FrozenNewRuntime(ScenarioRuntime):
    """Four/Ring runtime with canonical and Q16 RNG namespaces plus observers."""
    def __init__(self, name: str):
        super().__init__(name)
        self.output = OUT / name
        proposed_experiment_uid = uid("exp", {"path": str(OUT.resolve()), "scenario": name})
        with connect() as con:
            existing_experiment = con.execute(
                "SELECT experiment_uid FROM experiment WHERE name=? AND path=?",
                (EXPERIMENT, str(OUT.resolve()))).fetchone()
            self.experiment_uid = (existing_experiment[0] if existing_experiment
                                   else proposed_experiment_uid)
            con.execute("""INSERT OR IGNORE INTO experiment(
                        experiment_uid,name,path,protocol_hash,code_hash,metadata_json)
                        VALUES(?,?,?,?,?,?)""",
                        (self.experiment_uid, EXPERIMENT, str(OUT.resolve()),
                         content_hash({"K": K, "seeds": SEEDS, "tau": TAU_CRITIC}),
                         file_hash(Path(__file__)), canonical({"frozen_test": True})))
            con.commit()
        register_extra_controller(self, "hard_safety_q16", {
            "name": "frozen_generator_test_q16_v1", "root": FUTURE_ROOT,
            "derivation": "fold_in(fold_in(PRNGKey(root),state_token),future_index),then_step"})
        register_extra_controller(self, "orthoflow3_canonical", {
            "name": "frozen_stage1_evaluation_seed_v1", "evaluation_seed": FROZEN_EVAL_SEEDS[name],
            "derivation": "fold_in(PRNGKey(evaluation_seed),frozen_test_index),then_step"})

    def _root(self, state: dict[str, Any], future_index: int, chain: str):
        if chain in ("hard_safety", "orthoflow3_canonical"):
            return jax.random.fold_in(jax.random.PRNGKey(FROZEN_EVAL_SEEDS[self.name]), int(state["index"]))
        root = jax.random.fold_in(jax.random.PRNGKey(FUTURE_ROOT), state_token(state["uid"]))
        return jax.random.fold_in(root, int(future_index))

    def rollout(self, state: dict[str, Any], eta: np.ndarray, future_index: int,
                chain: str, *, save_trace: bool = False) -> dict[str, Any]:
        env = self.make_env(); self.reset(env, state); root = self._root(state, future_index, chain)
        use_eta = chain in ("orthoflow3", "orthoflow3_canonical")
        use_safety = chain != "no_safety"
        first_active=[]; second_active=[]; corrections=[]; ratios=[]; deform=[]
        min_wall=min_agent=min_obstacle=math.inf; positions=[env.positions.copy()]
        termination="timeout"; error=None
        try:
            for step in range(self.config.max_steps):
                flow=self.flow_world(env,jax.random.fold_in(root,step)); safe=flow
                if use_safety:
                    safe,_=self.project(env,flow); delta=float(np.linalg.norm(safe-flow))
                    first_active.append(delta>self.cbf.intervention_tol); corrections.append(delta)
                    ratios.append(delta/max(float(np.linalg.norm(flow)),1e-12))
                executed=safe
                if use_eta:
                    fields=self.basis.compute(env.positions,env.goals,safe,self.config.max_speed)
                    corrected=safe+fields.correction(eta); executed,_=self.project(env,corrected)
                    second_active.append(float(np.linalg.norm(executed-corrected))>self.cbf.intervention_tol)
                    deform.append(float(np.sum((executed-safe)**2)))
                _,_,done,info=env.step(executed); positions.append(env.positions.copy())
                walls,pairs=env.distances();min_wall=min(min_wall,float(np.min(walls)));min_agent=min(min_agent,float(np.min(pairs)))
                if self.kind=="ring":min_obstacle=min(min_obstacle,float(np.min(walls[:,0])))
                termination=str(info.get("termination","running"))
                if done:break
        except Exception as exc:
            # The frozen certified projector is the only expected numerical source.
            from shared_control.hard_projection import CBFSolverError
            if not isinstance(exc,(CBFSolverError,ValueError,FloatingPointError)):raise
            termination="numerical_failure";error=f"{type(exc).__name__}: {exc}"
        summary=env.summary(); collision=bool(summary.get("wall_collision",False) or summary.get("obstacle_collision",False) or summary.get("agent_collision",False))
        success=bool(summary.get("collision_free_success",False)) and termination=="success"
        path=np.asarray(positions); steps=int(summary.get("episode_steps",len(path)-1))
        dmean=float(np.mean(deform)) if deform else 0.; dint=float(self.config.dt*np.sum(deform)) if deform else 0.
        return {"scenario":self.name,"state_id":state["alias"],"state_uid":state["uid"],"eta":np.asarray(eta,float).tolist(),
          "future_index":int(future_index),"controller_chain":chain,"controller_uid":self.controllers[chain]["uid"],
          "outcome":"success" if success else ("collision" if collision else termination),"termination":termination,
          "success":success,"deadlock":termination=="deadlock","timeout":termination=="timeout","collision":collision,
          "numerical_failure":error is not None,"execution_error":error,"wall_collision":bool(summary.get("wall_collision",False)),
          "obstacle_collision":bool(summary.get("obstacle_collision",False)),"agent_collision":bool(summary.get("agent_collision",False)),
          "episode_length":steps,"minimum_wall_clearance":None if not np.isfinite(min_wall) else min_wall,
          "minimum_obstacle_clearance":None if not np.isfinite(min_obstacle) else min_obstacle,
          "minimum_agent_clearance":None if not np.isfinite(min_agent) else min_agent,"mode_signature":self.mode_fn(path),
          "projection_active_fraction":float(np.mean(first_active)) if first_active else 0.,
          "projection_correction_norm_mean":float(np.mean(corrections)) if corrections else 0.,
          "projection_correction_ratio_mean":float(np.mean(ratios)) if ratios else 0.,
          "second_projection_active_fraction":float(np.mean(second_active)) if second_active else 0.,
          "D_traj_mean":dmean,"D_traj_integrated":dint,"J_def":dint,
          "timestamp":datetime.now(timezone.utc).isoformat()}

    def conditioning(self, state: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
        env=self.make_env();self.reset(env,state);obs=np.asarray(self.observation(env),float)
        key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(state["uid"]))
        flow=self.flow_world(env,key)
        return np.concatenate((obs.reshape(-1),flow.reshape(-1))).tolist(),_environment_descriptor(self)

    def d_inst(self, state: dict[str, Any], eta: Iterable[float]) -> float:
        env=self.make_env();self.reset(env,state);key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(state["uid"]))
        flow=self.flow_world(env,key);safe,_=self.project(env,flow);fields=self.basis.compute(env.positions,env.goals,safe,self.config.max_speed)
        executed,_=self.project(env,safe+fields.correction(np.asarray(eta,float)))
        return float(np.sum((executed-safe)**2))


class DoubleFrozenRuntime:
    def __init__(self):
        self.name="double_bottleneck";self.output=OUT/self.name;self.output.mkdir(parents=True,exist_ok=True)
        self.datasets={name:db_old.FlowBC4ADataset(path,"all") for name,path in DOUBLE_POOLS.items()}
        first=next(iter(self.datasets.values()));self.policy,_=db_old.load_checkpoint(db_old.CHECKPOINT,first.environment_fingerprint)
        for path,expected in db_old.EXPECTED.items():
            if db_old.sha(path)!=expected:raise RuntimeError(f"Double frozen hash mismatch: {path}")
        initialize()
        with connect() as con:
            found=con.execute("SELECT scenario_uid FROM scenario WHERE name='DoubleBottleneck_4A'").fetchone()
            self.scenario_uid=found[0] if found else uid("scn",{"name":"DoubleBottleneck_4A","env":db_old.EXPECTED[ROOT/'double_bottleneck/environment.py']})
            con.execute("""INSERT OR IGNORE INTO scenario(
                        scenario_uid,name,code_config_fingerprint,metadata_json)
                        VALUES(?,?,?,?)""",
                        (self.scenario_uid,"DoubleBottleneck_4A",
                         db_old.EXPECTED[ROOT/'double_bottleneck/environment.py'],
                         canonical({"frozen":True})))
            self.experiment_uid=uid("exp",{"path":str(OUT.resolve()),"scenario":self.name})
            con.execute("""INSERT OR IGNORE INTO experiment(
                        experiment_uid,name,path,protocol_hash,code_hash,metadata_json)
                        VALUES(?,?,?,?,?,?)""",
                        (self.experiment_uid,EXPERIMENT,str(OUT.resolve()),
                         content_hash({"K":K,"seeds":SEEDS}),file_hash(Path(__file__)),
                         canonical({"frozen_test":True})))
            con.commit()
        self.states=self._states();self.checkpoint_sha=db_old.EXPECTED[db_old.CHECKPOINT]
        self.orthoflow_hash=file_hash(ROOT/"shared_control/basis_families.py");self.safety_hash=db_old.EXPECTED[ROOT/"shared_control/hard_projection.py"]
        self.controllers={}
        for chain,rng in {
          "hard_safety":{"name":"historical_matched_seed","canonical":True},
          "hard_safety_q16":{"name":"fixed_current_then_q16","root":DOUBLE_FUTURE_ROOT},
          "orthoflow3_canonical":{"name":"historical_matched_seed","canonical":True},
          "orthoflow3":{"name":"fixed_current_then_q16","root":DOUBLE_FUTURE_ROOT},}.items():
            payload={"scenario":self.name,"chain":chain,"flow_checkpoint_sha256":self.checkpoint_sha,
              "orthoflow3_sha256":self.orthoflow_hash if "orthoflow3" in chain else None,
              "safety_projection_sha256":self.safety_hash,"horizon":850,"dt":.05,
              "success_semantics":"frozen_double_bottleneck_v1","conditioning":"obs72+historical_current_raw_flow8",
              "rng":rng}
            self.controllers[chain]={"uid":uid("ctl",payload),"payload":payload}
        self._register()

    def _states(self):
        rows=[]
        for set_name in DOUBLE_POOLS:
            historical=load(DOUBLE_HARD/f"hard_safety_{set_name}_outcomes.json")["rollouts"]
            dataset=self.datasets[set_name]
            for r in historical:
                episode=dataset.by_family[r["family_id"]][0];physical={"positions":episode.initial_positions.tolist(),"velocities":episode.initial_velocities.tolist()}
                alias=f"DB_FROZEN_{set_name}_{int(r['rollout_id']):03d}";ch=content_hash({"physical":physical,"seed":r["seed"],"rollout_id":r["rollout_id"],"set":set_name})
                suid=uid("state",{"scenario":self.scenario_uid,"content":ch})
                key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(int(r["seed"])),int(r["rollout_id"])),0)
                raw=np.asarray(self.policy.sample_actions(db_old._initialize_env(db_old.Config(**dataset.config),episode).observation()[None],key)[0],float)
                obs=np.asarray(db_old._initialize_env(db_old.Config(**dataset.config),episode).observation(),float)
                rows.append({"uid":suid,"alias":alias,"index":len(rows),"content_hash":ch,"physical":physical,
                  "conditioning_flat":np.r_[obs.ravel(),raw.ravel()].tolist(),"environment_descriptor":dataset.config,
                  "metadata":{"set":set_name,"family_id":r["family_id"],"historical_seed":int(r["seed"]),"rollout_id":int(r["rollout_id"]),"historical":r}})
        return rows

    def _register(self):
        with connect() as con:
            for s in self.states:
                con.execute("""INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality)
                VALUES(?,?,?,?,?,?,?,?)""",(s["uid"],self.scenario_uid,"frozen_test",s["content_hash"],canonical(s["physical"]),canonical({}),canonical(s["metadata"]),"CONTENT_EXACT"))
                con.execute("INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)",(self.scenario_uid,s["alias"],s["uid"],self.experiment_uid))
            for c in self.controllers.values():
                p=c["payload"]
                con.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(c["uid"],self.scenario_uid,self.checkpoint_sha,p["orthoflow3_sha256"],self.safety_hash,"850","0.05",p["success_semantics"],p["conditioning"],canonical(p["rng"]),canonical(p),"EXACT_PROFILE"))
            con.commit()

    def _episode(self,state):
        m=state["metadata"];d=self.datasets[m["set"]];return d,d.by_family[m["family_id"]][0]

    def rollout(self,state,eta,future_index,chain,*,save_trace=False):
        dataset,episode=self._episode(state);m=state["metadata"];policy=self.policy
        canonical_chain=chain in ("hard_safety","orthoflow3_canonical");use_eta="orthoflow3" in chain
        current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(m["historical_seed"]),m["rollout_id"]),0)
        future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(DOUBLE_FUTURE_ROOT),state_token(state["uid"])),int(future_index))
        class Wrapped:
            def __init__(self):self.step=0
            def sample_actions(self,obs,unused):
                if canonical_chain:key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(m["historical_seed"]),m["rollout_id"]),self.step)
                else:key=current if self.step==0 else jax.random.fold_in(future,self.step)
                self.step+=1;return policy.sample_actions(obs,key)
        theta=np.asarray(eta,float) if use_eta else np.zeros(3)
        job={"job_id":f"{state['alias']}:{chain}:{future_index}:{eta_identity(theta)[0]}","stage":EXPERIMENT,"representation":"P1-OrthoFlow3","theta":theta.tolist(),"seed":DOUBLE_FUTURE_ROOT,"rollout_id":state_token(state["uid"]),"episode_id":state["alias"],"family_id":m["family_id"]}
        raw=run_db.run_one_with_jdef(Wrapped(),dataset,episode,job,3.303687238760696)
        valid=bool(raw.get("scientific_outcome_valid",False));term=str(raw.get("termination","numerical_failure"));collision=bool(raw.get("wall_collision") or raw.get("agent_collision"));steps=int(raw.get("episode_steps",0));integrated=float(raw.get("J_def",0.)) if use_eta else 0.
        return {"scenario":self.name,"state_id":state["alias"],"state_uid":state["uid"],"eta":np.asarray(eta,float).tolist(),"future_index":int(future_index),"controller_chain":chain,"controller_uid":self.controllers[chain]["uid"],"outcome":raw.get("outcome",term),"termination":term,"success":bool(raw.get("success")) and valid,"deadlock":"deadlock" in term,"timeout":term=="timeout","collision":collision,"numerical_failure":not valid,"wall_collision":bool(raw.get("wall_collision")),"agent_collision":bool(raw.get("agent_collision")),"episode_length":steps,"minimum_wall_clearance":raw.get("minimum_wall_clearance"),"minimum_agent_clearance":raw.get("minimum_agent_clearance"),"mode_signature":raw.get("coordination_mode"),"projection_active_fraction":None,"second_projection_active_fraction":None,"projection_removal_gt_half_fraction":raw.get("correction",{}).get("more_than_half_removed_fraction"),"D_traj_mean":integrated/max(dataset.config["dt"]*steps,1e-12),"D_traj_integrated":integrated,"J_def":integrated,"timestamp":datetime.now(timezone.utc).isoformat()}

    def conditioning(self,state):return state["conditioning_flat"],state["environment_descriptor"]

    def d_inst(self,state,eta):
        dataset,episode=self._episode(state);m=state["metadata"];env=db_old._initialize_env(db_old.Config(**dataset.config),episode);key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(m["historical_seed"]),m["rollout_id"]),0)
        raw=np.asarray(self.policy.sample_actions(env.observation()[None],key)[0],float);flow=db_old._radial_bound64(raw,dataset.config["max_speed"]);projector=CertifiedHardSafetyFilter();safe=np.asarray(projector(env.snapshot(),flow).velocity,float)
        g=db_old.correction("P1-OrthoFlow3",np.asarray(eta,float),env.positions,env.goals,safe,dataset.config["max_speed"],3.303687238760696);executed=np.asarray(projector(env.snapshot(),safe+g).velocity,float)
        return float(np.sum((executed-safe)**2))


def runtimes():
    return {"double_bottleneck":DoubleFrozenRuntime(),"four_way_intersection":FrozenNewRuntime("four_way_intersection"),"ring_exchange":FrozenNewRuntime("ring_exchange")}


def context_vector(env: dict[str,Any], norm: dict[str,Any], scenario: str) -> np.ndarray:
    values=scalar_environment_fields(env);keys=norm["environment_keys"];c=np.zeros(len(keys)+len(SCENARIOS),np.float32)
    for j,k in enumerate(keys):c[j]=values.get(k,0.)
    c[len(keys)+SCENARIOS.index(scenario)]=1.;n=norm["scenarios"][scenario]
    return (c-np.asarray(n["c_mean"],np.float32))/np.asarray(n["c_std"],np.float32)


def load_models(norm):
    dims={s:len(norm["scenarios"][s]["h_mean"]) for s in SCENARIOS};cdim=len(norm["environment_keys"])+len(SCENARIOS)
    gm=Generator();gp=serialization.from_bytes(merge_initialized(gm,dims,cdim),GENERATOR_CKPT.read_bytes())
    cm=Critic();cp=serialization.from_bytes(merge_initialized(cm,dims,cdim,critic=True),CRITIC_CKPT.read_bytes())
    return gm,gp,cm,cp


def freeze() -> dict[str,Any]:
    OUT.mkdir(parents=True,exist_ok=True);rt=runtimes();norm=load(NORMALIZATION);gm,gp,cm,cp=load_models(norm);states=[]
    for scenario in SCENARIOS:
        runtime=rt[scenario];n=norm["scenarios"][scenario]
        for state in runtime.states:
            flat,env=runtime.conditioning(state);h=(np.asarray(flat,np.float32)-np.asarray(n["h_mean"],np.float32))/np.asarray(n["h_std"],np.float32);c=context_vector(env,norm,scenario)
            raw=np.asarray(gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=getattr(gm,METHOD[scenario])))[0]
            mean=np.asarray(eta_mean(jnp.asarray(raw)),float);rng=np.random.default_rng(stable_int("generator-v1-proposals",GENERATOR_SEED,scenario,state["uid"]));samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(K,0),jnp.asarray(rng.standard_normal((K,3)),jnp.float32)),float);etas=np.asarray([mean,*samples]);hh=np.repeat(h[None],len(etas),0);cc=np.repeat(c[None],len(etas),0);ee=(etas-CENTER)/RADIUS
            scores=sigmoid(np.asarray(cm.apply(cp,jnp.asarray(hh),jnp.asarray(cc),jnp.asarray(ee),method=getattr(cm,METHOD[scenario]))));dinst=[runtime.d_inst(state,e) for e in etas];critic=int(np.argmax(scores));feasible=[i for i,x in enumerate(scores) if x>=TAU_CRITIC];mindef=min(feasible,key=lambda i:(dinst[i],i)) if feasible else -1
            states.append({"scenario":scenario,"state_uid":state["uid"],"state_id":state["alias"],"proposal_seed":stable_int("generator-v1-proposals",GENERATOR_SEED,scenario,state["uid"]),"generator_mean":mean.tolist(),"samples":samples.tolist(),"critic_scores":scores.tolist(),"critic_index":critic,"min_def_index":mindef,"D_inst_proposals":dinst,"sigma":np.asarray(0.025+0.275*jax.nn.sigmoid(jnp.asarray(raw[3:])),float).tolist()})
    manifest={"schema":"orthoflow3_generator_critic_frozen_test_proposals_v1","created_before_outcome_query":True,"states":states,"state_counts":dict(Counter(r["scenario"] for r in states)),"K":K,"proposal_set":"transformed generator mean + 4 deterministic stochastic samples","proposal_seed_rule":"SHA256(generator-v1-proposals|41|scenario|state_uid)","tau_critic":TAU_CRITIC,"tau_source":"critic predicts empirical Q; frozen mathematical 15/16 threshold","fixed_scenario_eta":FIXED_ETA}
    dump("frozen_proposals.json",manifest)
    artifacts={str(p.relative_to(ROOT)):file_hash(p) for p in [GENERATOR_CKPT,CRITIC_CKPT,NORMALIZATION,LEARN/"train_evaluate.py",ROOT/"shared_control/basis_families.py",ROOT/"shared_control/hard_projection.py",ROOT/"new_benchmark_common/safety_eta3.py",ROOT/"double_bottleneck/environment.py",ROOT/"four_way_intersection/environment.py",ROOT/"ring_exchange/environment.py"]}
    dump("frozen_artifacts.json",{"artifacts":artifacts,"eta_domain":{"low":LOW.tolist(),"high":HIGH.tolist()},"Q16_seeds":list(SEEDS),"models_modified":False,"proposal_manifest_sha256":file_hash(OUT/"frozen_proposals.json")})
    return manifest


def unique_candidates(row):
    values=[("B0",[0.,0.,0.]),("B1",FIXED_ETA[row["scenario"]]),("mean",row["generator_mean"]),*[ (f"sample_{i}",e) for i,e in enumerate(row["samples"]) ]]
    out=[];seen=set()
    for name,e in values:
        key=eta_identity(e)[0]
        if key not in seen:out.append((name,e));seen.add(key)
    return out


def seed_keys(seeds):return [canonical({"future_index":int(s)}) for s in seeds]


def lookup(runtime,state,eta,chain,seeds):
    keys=seed_keys(seeds);eu=eta_identity(eta)[0]
    with connect(True) as con:x=lookup_exact(con,state["uid"],eu,runtime.controllers[chain]["uid"],keys)
    by={r["seed_key"]:dict(r) for r in x["records"]};return {s:by[k] for s,k in zip(seeds,keys) if k in by},[s for s,k in zip(seeds,keys) if k in set(x["missing_seeds"])]


def plan_requests(rt,manifest):
    by={(r["scenario"],r["state_uid"]):r for r in manifest["states"]}
    req=[]
    for scenario,runtime in rt.items():
        for state in runtime.states:
            row=by[(scenario,state["uid"])]
            # B0 uses one projection; all nonzero proposal candidates use the second projection.
            req.append((runtime,state,"B0",[0.,0.,0.],"hard_safety",[0],"canonical"))
            req.append((runtime,state,"B0",[0.,0.,0.],"hard_safety_q16",SEEDS,"q16"))
            for name,eta in unique_candidates(row):
                if name=="B0":continue
                req.append((runtime,state,name,eta,"orthoflow3_canonical",[0],"canonical"))
                req.append((runtime,state,name,eta,"orthoflow3",SEEDS,"q16"))
    return req


def preflight() -> dict[str,Any]:
    manifest=load("frozen_proposals.json");rt=runtimes();detail=[];total=Counter()
    for runtime,state,name,eta,chain,seeds,view in plan_requests(rt,manifest):
        rows,missing=lookup(runtime,state,eta,chain,seeds);detail.append({"scenario":runtime.name,"state_uid":state["uid"],"candidate":name,"chain":chain,"view":view,"requested":len(seeds),"reused":len(rows),"missing":missing});total["requested"]+=len(seeds);total["reused"]+=len(rows);total["missing"]+=len(missing)
    out={"database":str(DB_PATH),"summary":dict(total),"details":detail,"proposal_manifest_sha256":file_hash(OUT/"frozen_proposals.json")}
    current=OUT/"cache_preflight.json";initial=OUT/"initial_cache_preflight.json"
    if current.exists() and not initial.exists():shutil.copy2(current,initial)
    dump("cache_preflight.json",out);return out


def insert_rollout(runtime,row,raw_path):
    raw_path.parent.mkdir(parents=True,exist_ok=True)
    with raw_path.open("a") as f:f.write(canonical(row)+"\n");f.flush()
    eu,eta,hx=eta_identity(row["eta"]);sk=canonical({"future_index":int(row["future_index"])});rid=uid("roll",{"state":row["state_uid"],"eta":eu,"controller":row["controller_uid"],"seed":sk});rh=content_hash(row);src=uid("src",{"path":str(raw_path.resolve())})
    with connect() as con:
        con.execute("""INSERT OR IGNORE INTO eta(
                    eta_uid,eta1,eta2,eta3,normalized_eta_json,canonical_hex)
                    VALUES(?,?,?,?,?,?)""",
                    (eu,*eta,canonical(normalized_eta(eta).tolist()),hx))
        con.execute("INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)",(src,runtime.experiment_uid,str(raw_path.resolve()),"LIVE",".jsonl","SEED_EXACT",0));con.execute("UPDATE source_file SET rows_seen=rows_seen+1 WHERE source_uid=?",(src,))
        old=con.execute("SELECT success,deadlock,timeout,collision,numerical_failure FROM rollout WHERE rollout_uid=?",(rid,)).fetchone();core=(int(row["success"]),int(row["deadlock"]),int(row["timeout"]),int(row["collision"]),int(row["numerical_failure"]))
        if old is not None and tuple(old)!=core:raise RuntimeError(f"DB conflict {rid}")
        con.execute("""INSERT OR IGNORE INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,original_source_file,timestamp,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(rid,row["state_uid"],eu,row["controller_uid"],sk,sk,*core,row["episode_length"],row.get("J_def"),row.get("minimum_wall_clearance"),row.get("minimum_agent_clearance"),row["outcome"],runtime.experiment_uid,str(raw_path.resolve()),row["timestamp"],"EXACT_REUSE",rh))
        con.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)",(rid,src,None));con.commit()


def run(shard:int,shards:int):
    manifest=load("frozen_proposals.json");rt=runtimes();requests=plan_requests(rt,manifest)
    # Shard the atomic seed tuple, not a variable-size method group.  This is
    # a scheduling-only transformation and keeps the globally fixed seed order
    # and every scientific identity unchanged.
    tasks=[(runtime,state,name,eta,chain,seed,view)
           for runtime,state,name,eta,chain,seeds,view in requests for seed in seeds]
    selected=[r for i,r in enumerate(tasks) if i%shards==shard]
    raw=OUT/"raw"/f"atomic_shard{shard}of{shards}.jsonl";physical=reused=0
    for i,(runtime,state,name,eta,chain,seed,view) in enumerate(selected):
        rows,missing=lookup(runtime,state,eta,chain,[seed])
        if not missing:reused+=1;continue
        result=None
        for attempt in range(4):
            result=runtime.rollout(state,np.asarray(eta,float),seed,chain);result.update({"candidate":name,"view":view,"attempt":attempt,"shard":shard})
            insert_rollout(runtime,result,raw);physical+=1
            if not result["numerical_failure"]:break
        if (i+1)%100==0:print(json.dumps({"shard":shard,"atomic_tasks":i+1,"physical":physical,"reused":reused}),flush=True)
    out={"shard":shard,"shards":shards,"atomic_tasks":len(selected),"physical":physical,"reused":reused};dump(f"run_atomic_shard{shard}of{shards}.json",out);return out


def raw_metrics():
    result={}
    for p in sorted((OUT/"raw").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);result[(r["state_uid"],r["controller_uid"],eta_identity(r["eta"])[0],int(r["future_index"]))]=r
    return result


def collect(runtime,state,eta,chain,seeds,metrics):
    rows,missing=lookup(runtime,state,eta,chain,seeds)
    if missing:
        eu=eta_identity(eta)[0];keys=seed_keys(missing)
        with connect(True) as con:
            q=",".join("?"*len(keys));invalid=con.execute(
                f"""SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
                AND numerical_failure=1 AND seed_key IN ({q})""",
                (state["uid"],eu,runtime.controllers[chain]["uid"],*keys)).fetchall()
        bykey={r["seed_key"]:dict(r) for r in invalid}
        for seed,key in zip(missing,keys):
            if key in bykey:rows[seed]=bykey[key]
        unresolved=[seed for seed in missing if seed not in rows]
        if unresolved:raise RuntimeError(f"missing outcomes {runtime.name} {state['uid']} {chain} {unresolved}")
    out=[]
    for seed in seeds:
        db=dict(rows[seed]);key=(state["uid"],runtime.controllers[chain]["uid"],eta_identity(eta)[0],seed);detail=metrics.get(key,{})
        out.append({**db,**{k:detail.get(k,db.get(k)) for k in ("wall_collision","obstacle_collision","agent_collision","D_traj_mean","D_traj_integrated","projection_active_fraction","second_projection_active_fraction","mode_signature")}})
    return out


def finalize():
    manifest=load("frozen_proposals.json");rt=runtimes();metrics=raw_metrics();by={(r["scenario"],r["state_uid"]):r for r in manifest["states"]};per=[];qmat=[]
    methods=("B0 Hard Safety","B1 Fixed eta","B2 Generator Mean","B3 Random Proposal","B4 Oracle finite proposals","B5 Critic-selected","Min-Def Critic-Feasible","Oracle Min-Dinst","Oracle Min-Dtraj")
    for scenario,runtime in rt.items():
        for state in runtime.states:
            p=by[(scenario,state["uid"])];cand=[("mean",p["generator_mean"]),*[ (f"sample_{i}",e) for i,e in enumerate(p["samples"]) ]];evidence={}
            b0q=collect(runtime,state,[0,0,0],"hard_safety_q16",SEEDS,metrics);b0c=collect(runtime,state,[0,0,0],"hard_safety",[0],metrics)
            evidence["B0 Hard Safety"]=(b0q,b0c,[0,0,0],0.,"B0")
            b1q=collect(runtime,state,FIXED_ETA[scenario],"orthoflow3",SEEDS,metrics);b1c=collect(runtime,state,FIXED_ETA[scenario],"orthoflow3_canonical",[0],metrics);b1d=runtime.d_inst(state,FIXED_ETA[scenario]);evidence["B1 Fixed eta"]=(b1q,b1c,FIXED_ETA[scenario],b1d,"B1")
            ce={}
            for i,(name,eta) in enumerate(cand):
                ce[name]=(collect(runtime,state,eta,"orthoflow3",SEEDS,metrics),collect(runtime,state,eta,"orthoflow3_canonical",[0],metrics),eta,p["D_inst_proposals"][i],name)
            def kval(item):return sum(int(r["success"]) for r in item[1][0])
            oracle=max(ce.items(),key=lambda z:(kval(z),-cand.index(next(x for x in cand if x[0]==z[0]))))[0]
            robust_names=[name for name,x in ce.items() if kval((name,x))>=15]
            odinst=min(robust_names,key=lambda n:(ce[n][3],cand.index(next(x for x in cand if x[0]==n)))) if robust_names else None
            def mdtraj(n):return np.mean([r.get("D_traj_mean") for r in ce[n][0] if r.get("D_traj_mean") is not None])
            odtraj=min(robust_names,key=lambda n:(mdtraj(n),cand.index(next(x for x in cand if x[0]==n)))) if robust_names else None
            critic=cand[p["critic_index"]][0];mindef=(cand[p["min_def_index"]][0] if p["min_def_index"]>=0 else None)
            evidence["B2 Generator Mean"]=ce["mean"];evidence["B3 Random Proposal"]=ce["sample_0"];evidence["B4 Oracle finite proposals"]=ce[oracle];evidence["B5 Critic-selected"]=ce[critic]
            evidence["Min-Def Critic-Feasible"]=ce[mindef] if mindef else evidence["B0 Hard Safety"]
            evidence["Oracle Min-Dinst"]=ce[odinst] if odinst else evidence["B0 Hard Safety"]
            evidence["Oracle Min-Dtraj"]=ce[odtraj] if odtraj else evidence["B0 Hard Safety"]
            def robust_status(rows):
                succ=sum(int(r["success"]) for r in rows)
                valid=[r for r in rows if not r.get("numerical_failure")]
                failures=len(valid)-succ
                return True if succ>=15 else False if failures>=2 else None
            base_rob=robust_status(b0q);base_single=(None if b0c[0].get("numerical_failure") else bool(b0c[0]["success"]))
            state_row={"scenario":scenario,"state_uid":state["uid"],"state_id":state["alias"],"critic_index":p["critic_index"],"oracle_candidate":oracle,"critic_candidate":critic,"critic_scores":p["critic_scores"],"min_def_candidate":mindef}
            for method in methods:
                qrows,crows,eta,dinst,source=evidence[method];succ=sum(int(r["success"]) for r in qrows);rob=robust_status(qrows);single=(None if crows[0].get("numerical_failure") else bool(crows[0]["success"]));dvals=[r.get("D_traj_mean") for r in qrows if r.get("D_traj_mean") is not None]
                # The legacy Double-Bottleneck adapter does not expose a
                # certified second-projection activity trace.  Do not mix
                # stale diagnostic fields from superseded/resumed records
                # into the frozen result; report this metric as unavailable.
                projection=[] if scenario=="double_bottleneck" else [r.get("second_projection_active_fraction") for r in qrows if r.get("second_projection_active_fraction") is not None]
                rec={"scenario":scenario,"state_uid":state["uid"],"method":method,"source_candidate":source,"eta":eta,"Q16_lower":succ/16,"Q16":succ/16 if not any(r.get("numerical_failure") for r in qrows) else None,"successes":succ,"numerical_seeds":sum(bool(r.get("numerical_failure")) for r in qrows),"robust":rob,"single_success":single,"single_outcome":outcome_class(crows[0]),"D_inst":dinst,"D_traj_mean":float(np.mean(dvals)) if dvals else 0.,"D_traj_integrated_mean":float(np.mean([r.get("D_traj_integrated",0.) or 0. for r in qrows])),"projection_activity":float(np.mean(projection)) if projection else None,"episode_length_mean":float(np.mean([r["episode_length"] for r in qrows])),"rescue_robust":bool(rob is True and base_rob is False),"break_robust":bool(base_rob is True and rob is False),"rescue_single":bool(single is True and base_single is False),"break_single":bool(base_single is True and single is False),"terminal_counts":dict(Counter(outcome_class(r) for r in qrows))}
                qmat.append(rec);state_row[method]=rec
            per.append(state_row)
    dump("per_state_method_outcomes.json",per);dump("q16_matrices.json",qmat)
    summary={};conversion={};exploitation={};pareto={}
    for sc in SCENARIOS:
        rr=[r for r in qmat if r["scenario"]==sc];summary[sc]={}
        for m in methods:
            z=[r for r in rr if r["method"]==m];exactq=[r["Q16"] for r in z if r["Q16"] is not None];summary[sc][m]={"states":len(z),"single_seed_success":sum(r["single_success"] is True for r in z),"single_seed_numerical":sum(r["single_success"] is None for r in z),"robust_success":sum(r["robust"] is True for r in z),"robust_non_success":sum(r["robust"] is False for r in z),"robust_unresolved":sum(r["robust"] is None for r in z),"rescue_single":sum(r["rescue_single"] for r in z),"break_single":sum(r["break_single"] for r in z),"rescue_robust":sum(r["rescue_robust"] for r in z),"break_robust":sum(r["break_robust"] for r in z),"mean_Q16":float(np.mean(exactq)) if exactq else None,"mean_Q16_lower_all":float(np.mean([r["Q16_lower"] for r in z])),"D_inst":stats(r["D_inst"] for r in z),"D_traj":stats(r["D_traj_mean"] for r in z),"eta_norm":stats(np.linalg.norm(r["eta"]) for r in z),"projection_activity":stats(r["projection_activity"] for r in z if r["projection_activity"] is not None),"completion_steps":stats(r["episode_length_mean"] for r in z)}
        ss=[r for r in per if r["scenario"]==sc];conversion[sc]=dict(Counter(f"{r['B0 Hard Safety']['single_outcome']}->{r['B5 Critic-selected']['single_outcome']}" for r in ss))
        bad=[]
        for r in ss:
            b4=r["B4 Oracle finite proposals"];b5=r["B5 Critic-selected"]
            if b4["robust"] is True and b5["robust"] is False:bad.append({"state_uid":r["state_uid"],"critic_scores":r["critic_scores"],"oracle":r["oracle_candidate"],"selected":r["critic_candidate"],"oracle_Q16":b4["Q16"],"selected_Q16":b5["Q16"]})
        unresolved=[{"state_uid":r["state_uid"],"oracle":r["oracle_candidate"],"selected":r["critic_candidate"],"selected_successes":r["B5 Critic-selected"]["successes"],"numerical_seeds":r["B5 Critic-selected"]["numerical_seeds"]} for r in ss if r["B4 Oracle finite proposals"]["robust"] is True and r["B5 Critic-selected"]["robust"] is None]
        exploitation[sc]={"exact_same_candidate":sum(r["oracle_candidate"]==r["critic_candidate"] for r in ss),"states":len(ss),"robust_state_gap":summary[sc]["B4 Oracle finite proposals"]["robust_success"]-summary[sc]["B5 Critic-selected"]["robust_success"],"mean_Q16_gap":summary[sc]["B4 Oracle finite proposals"]["mean_Q16"]-summary[sc]["B5 Critic-selected"]["mean_Q16"],"oracle_robust_critic_nonrobust":len(bad),"oracle_robust_critic_unresolved":len(unresolved),"cases":bad,"unresolved_cases":unresolved}
        pareto[sc]={"states_with_lower_Dinst_robust_than_B5":sum(r["Oracle Min-Dinst"]["robust"] and r["Oracle Min-Dinst"]["D_inst"]+1e-12<r["B5 Critic-selected"]["D_inst"] for r in ss),"states_with_lower_Dtraj_robust_than_B5":sum(r["Oracle Min-Dtraj"]["robust"] and r["Oracle Min-Dtraj"]["D_traj_mean"]+1e-12<r["B5 Critic-selected"]["D_traj_mean"] for r in ss)}
    dump("summary.json",summary);dump("conversion_matrices.json",conversion);dump("critic_exploitation_audit.json",exploitation);dump("pareto_analysis.json",pareto)
    with connect(True) as con:
        integrity={"integrity_check":con.execute("PRAGMA integrity_check").fetchone()[0],"foreign_key_violations":[dict(r) for r in con.execute("PRAGMA foreign_key_check")],"rollouts":con.execute("SELECT COUNT(*) FROM rollout").fetchone()[0]}
    dump("database_integrity.json",integrity);post=preflight();dump("cache_postflight.json",post)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("freeze","preflight","run","finalize"));p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=1);a=p.parse_args()
    value=freeze() if a.stage=="freeze" else preflight() if a.stage=="preflight" else run(a.shard,a.shards) if a.stage=="run" else finalize();print(json.dumps(value if a.stage!="freeze" else {"states":len(value["states"]),"counts":value["state_counts"]},indent=2))

if __name__=="__main__":main()
