"""Fill the measured terminal-state coverage gap with 12 real teacher episodes."""
from . import resources  # noqa: F401
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import canonical,connect,eta_identity,lookup_exact,transaction,uid
from . import pilot
from .experiment import dump,sha


def main():
    folder=Path("diagnostics/drone_3d_pipeline_v1/terminal_teacher"); folder.mkdir(exist_ok=True,parents=True)
    old=json.loads(Path("diagnostics/drone_3d_pilot_v3/protocol.json").read_text())
    scid=old["scenario_uid"]; config=asdict(pilot.Config())
    # This is the original pilot goal controller and simulator, not Flow.
    cid=old["controllers"]["goal_cbf"]["uid"]
    eid=eta_identity([0.,0.,0.])[0]; states=[]
    for i in range(12):
        seed=70000+i; rng=np.random.default_rng(seed)
        goals=np.array(pilot.initial_state([17,23,41][i%3])["goals"])
        p=goals+rng.uniform(-1,1,(2,3))*[.4,.07,.12]
        if i in (0,8):p=goals.copy()  # real immediate-success zero-action episodes
        physical=dict(positions=p.tolist(),velocities=np.zeros((2,3)).tolist(),goals=goals.tolist())
        sid=uid("state",dict(scenario=scid,physical=physical)); key=canonical(dict(seed=seed))
        rid=uid("roll",dict(state=sid,eta=eid,controller=cid,seed=key))
        states.append(dict(uid=sid,physical=physical,seed=seed,seed_key=key,rollout_uid=rid,split="train" if i<8 else "dev"))
    exp=uid("exp",dict(path=str(folder.resolve())))
    manifest=dict(requests=[dict(state_uid=s["uid"],controller_uid=cid,eta_uid=eid,seed_keys=[s["seed_key"]]) for s in states])
    protocol=dict(states=states,controller_uid=cid,scenario_uid=scid,experiment_uid=exp,source_sha256=sha(pilot.__file__),
                  history="Reused original pilot goal controller and three route demos. Cached Flow eta probes cross corridor but lack terminal precision; zero success labels, so Q training is gated.",
                  remaining_gap="terminal-neighborhood and at-goal expert support; 12 true goal+CBF episodes",changes_to_success=False)
    if (folder/"protocol.json").exists():
        assert json.loads((folder/"protocol.json").read_text())==protocol
    else:dump(folder/"protocol.json",protocol)
    dump(folder/"planned_rollouts.json",manifest)
    with connect() as db,transaction(db):
        db.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)",
                   (exp,"Drone3D terminal teacher coverage",str(folder.resolve()),sha(folder/"protocol.json"),sha(__file__),canonical(dict(max_rollouts=12))))
        for s in states:
            db.execute("INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)",
                       (s["uid"],scid,s["split"],uid("physical",s["physical"]),canonical(s["physical"]),canonical(dict(step=0,monitor_history=[])),canonical(dict(goals=s["physical"]["goals"])),canonical(dict(seed=s["seed"])),"CONTENT_EXACT"))
    dump(folder/"cache_preflight.json",preflight(folder/"planned_rollouts.json"))
    rows=[]; new=0; start=time.perf_counter()
    for s in states:
        dest=folder/f"seed{s['seed']}.json"; trace_path=dest.with_suffix(".npz")
        with connect(True) as db:hit=lookup_exact(db,s["uid"],eid,cid,[s["seed_key"]])
        if dest.exists():
            r=json.loads(dest.read_text()); assert sha(trace_path)==r["trace_sha256"]
        elif hit["status"]!="INCOMPATIBLE":raise RuntimeError("retrieve original cached artifact; do not rerun")
        else:
            trace,r=pilot.rollout(s["physical"],"goal_cbf",pilot.Config())
            r.update(cache_identity_schema="registered_exact_uid_v1",scenario_uid=scid,state_uid=s["uid"],eta_uid=eid,controller_uid=cid,
                     eta=[0.,0.,0.],rollout_uid=s["rollout_uid"],seed=s["seed"],split=s["split"],flow_checkpoint_sha256=None,
                     initial_positions=s["physical"]["positions"],initial_velocities=s["physical"]["velocities"],goals=s["physical"]["goals"],trace_path=str(trace_path.resolve()))
            np.savez_compressed(trace_path,**trace,metadata_json=np.array(json.dumps(r))); r["trace_sha256"]=sha(trace_path)
            dump(dest,r);append_journal([r],exp,"terminal_cpu1");new+=1
        rows.append(r)
    subprocess.run([sys.executable,"-m","shared_rollout_db.src.merger","--experiment",exp],check=True)
    report=dict(new_rollouts=new,successes=sum(r["success"] for r in rows),episodes=len(rows),wall_seconds=time.perf_counter()-start,
                transitions=sum(r["episode_length"] for r in rows),records=[str((folder/f"seed{s['seed']}.json").resolve()) for s in states])
    dump(folder/"summary.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
