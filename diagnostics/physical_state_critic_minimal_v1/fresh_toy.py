"""Prospective selected-only Q16 Toy FF confirmation on 48 frozen new states."""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import EtaOnly as WideEtaOnly
from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.physical_state_critic_formal_v1 import loso_worker
from diagnostics.physical_state_critic_formal_v1.merge_loso import merge as merge_formal
from diagnostics.physical_state_critic_minimal_v1.pilot import ROOT, HERE, bootstrap, read, sha, write
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import eta_identity

FORMAL = ROOT / "diagnostics/physical_state_critic_formal_v1"
PANEL = FORMAL / "loso/toy_give_way/test"
DEST = HERE / "fresh_toy_48"
N = 48
METHODS = ("within_state", "within_eta_wide", "within_state_shuffled",
           "loso_state", "loso_eta_wide", "Safety")


def load_panel():
    state_file = read(FORMAL/"states_toy_give_way.json")
    states = {s["alias"]:s for s in state_file["states"] if s["split"]=="test"}
    proposal = read(PANEL/"proposals.json")["states"][:N]
    assert len(proposal)==N
    rows=[dict(scene="toy_give_way",physical=states[p["state_alias"]]["physical"]) for p in proposal]
    ids=[p["state_uid"] for p in proposal]
    assert len(set(ids))==N
    return ids,proposal,states,rows


def score(panel,kind,x,raw_eta,shuffled=None):
    folder=HERE/"strict"/panel/kind
    info=read(folder/"fit.json")
    assert info["target_labels_opened"] is False
    ck=folder/"checkpoint.msgpack";assert sha(ck)==info["checkpoint_sha256"]
    e=jnp.asarray((raw_eta-np.asarray(info["eta_center"])) / np.asarray(info["eta_radius"]))
    net=rep.Critic() if kind=="state" else WideEtaOnly()
    xx={k:jnp.asarray(v) for k,v in x.items()}
    get=lambda ii:{k:v[ii] for k,v in xx.items()}
    template=net.init(jax.random.PRNGKey(17),get(jnp.array([0])),e[:1,:1].reshape(1,3))
    params=serialization.from_bytes(template,ck.read_bytes())
    flat=e.reshape(-1,3);indices=np.repeat(np.arange(N),16)
    if shuffled is not None:indices=shuffled[indices]
    values=[]
    for a in range(0,len(flat),256):
        ii=jnp.asarray(indices[a:a+256]);values.append(np.asarray(net.apply(params,get(ii),flat[a:a+256])))
    logits=np.concatenate(values).reshape(N,16)
    assert np.isfinite(logits).all()
    return logits,sha(ck)


def freeze():
    if (DEST/"choices.json").exists():return read(DEST/"choices.json")
    ids,proposals,_,rows=load_panel()
    raw_eta=np.asarray([p["eta"] for p in proposals],np.float32)
    assert raw_eta.shape==(N,16,3)
    x=physical_entities.batch(rows)
    assert np.all(x["agents"][...,3:7]==0) and np.all(x["globals"]==0)
    rng=np.random.default_rng(20261008);perm=rng.permutation(N)
    while np.any(perm==np.arange(N)):perm=rng.permutation(N)
    spec=(('within_state','within_toy','state',None),
          ('within_eta_wide','within_toy','eta_only_wide',None),
          ('within_state_shuffled','within_toy','state',perm),
          ('loso_state','loso_toy','state',None),
          ('loso_eta_wide','loso_toy','eta_only_wide',None))
    logits={};hashes={}
    for method,panel,kind,p in spec:
        logits[method],hashes[method]=score(panel,kind,x,raw_eta,p)
    choices={method:np.argmax(values,axis=1).tolist() for method,values in logits.items()}
    choices['Safety']=[16]*N
    cases=read(PANEL/"cases.json")
    by=defaultdict(list)
    for c in cases:
        if c['state_index']<N:by[(c['state_index'],c['eta_index'])].append(c)
    selected=[];requests=[]
    for i in range(N):
        for j in sorted({choices[m][i] for m in METHODS}):
            a=sorted(by[(i,j)],key=lambda c:c['future_index'])
            assert len(a)==16 and [c['future_index'] for c in a]==list(range(16))
            selected+=a
            requests.append(dict(state_uid=a[0]['state_uid'],eta_uid=a[0]['eta_uid'],
                                 controller_uid=a[0]['controller_uid'],seed_keys=[c['seed_key'] for c in a]))
    DEST.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DEST/"predictions.npz",state_uids=np.asarray(ids),eta=raw_eta,
                        **{k:v for k,v in logits.items()})
    assert not (PANEL/"outcomes.npz").exists(),"formal full TEST outcome unexpectedly opened"
    frozen=dict(schema="fresh_toy48_selected_only_v1",n_states=N,source_state_file_sha256=sha(FORMAL/"states_toy_give_way.json"),
                proposal_sha256=sha(PANEL/"proposals.json"),full_case_manifest_sha256=sha(PANEL/"cases.json"),
                state_uids=ids,choice_indices=choices,shuffle_permutation=perm.tolist(),
                checkpoint_sha256=hashes,candidates_per_state=16,seed_count_per_selected_eta=16,
                selected_eta_state_pairs=len(requests),selected_seed_rollouts=len(selected),
                test_labels_opened=False,oracle_not_estimated=True)
    write(DEST/"choices.json",frozen)
    write(DEST/"selected_cases.json",selected)
    write(DEST/"planned_rollouts.json",dict(requests=requests,execute_missing=True))
    print(json.dumps(dict(states=N,pairs=len(requests),seeds=len(selected),unique_choices={m:len(set(v)) for m,v in choices.items()})))
    return frozen


def plan():
    import subprocess,sys
    frozen=read(DEST/"choices.json")
    assert frozen["test_labels_opened"] is False
    subprocess.run([sys.executable,"-m","shared_rollout_db.plan","--manifest",str(DEST/"planned_rollouts.json"),
                    "--output",str(DEST/"cache_preflight.json")],cwd=ROOT,check=True)
    selected=read(DEST/"selected_cases.json")
    con=sqlite3.connect(f"file:{ROOT/'shared_rollout_db/rollout.sqlite'}?mode=ro",uri=True)
    missing=[];seen=0
    for c in selected:
        row=con.execute("SELECT numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?",(c['rollout_uid'],)).fetchone()
        if row is None:missing.append(c)
        else:
            assert row[1]==0
            seen+=1
    con.close()
    write(DEST/"missing_cases.json",missing)
    items=[dict(start=i,end=min(i+128,len(missing))) for i in range(0,len(missing),128)]
    write(DEST/"work_items.json",items)
    print(json.dumps(dict(selected=len(selected),exact_present=seen,missing=len(missing),shards=len(items))))


def run(index):
    item=read(DEST/"work_items.json")[index]
    cases=read(DEST/"missing_cases.json")[item['start']:item['end']]
    frozen=read(DEST/"choices.json");assert frozen["test_labels_opened"] is False
    ids,proposals,states,_=load_panel()
    runtime=loso_worker.ToyRuntime()
    records=[];new=recovered=0
    for c in cases:
        assert c['state_uid']==ids[c['state_index']]
        prop=proposals[c['state_index']];source=states[prop['state_alias']]
        record_path=PANEL/"raw"/(c['rollout_uid']+'.json')
        if record_path.exists():
            record=read(record_path);assert record['case']==c;recovered+=1
        else:
            raw=runtime.rollout(dict(proposal=prop,source=source,physical=source['physical']),
                                np.asarray(c['eta'],float),c['future_index'],c['eta_index']==16)
            assert raw['state_uid']==c['state_uid']
            assert eta_identity(raw['eta'])[0]==c['eta_uid']
            numerical=int(raw['numerical_failure']);success=int(raw['success']);collision=int(raw['collision'])
            assert not(success and (numerical or collision))
            record=dict(schema='formal_q_loso_seed_v1',experiment_uid=loso_worker.EXPERIMENT,
                        case=c,success=success,deadlock=int(raw['deadlock']),timeout=int(raw['timeout']),
                        collision=collision,numerical_failure=numerical,
                        episode_length=raw['episode_length'],outcome=raw['outcome'],raw=raw,
                        protocol_sha256=sha(FORMAL/"loso_protocol.json"))
            loso_worker.write(record_path,record);new+=1
        records.append(record)
    if records:append_journal(records,loso_worker.EXPERIMENT,f'fresh_toy48_{index:04d}')
    write(DEST/"receipts"/f"{index:04d}.json",dict(index=index,cases=len(cases),new=new,recovered=recovered,complete=True))
    print(json.dumps(dict(index=index,cases=len(cases),new=new,recovered=recovered)))


def evaluate():
    items=read(DEST/"work_items.json")
    assert all((DEST/"receipts"/f"{i:04d}.json").exists() for i in range(len(items)))
    merge_formal()
    frozen=read(DEST/"choices.json")
    cases=read(DEST/"selected_cases.json")
    con=sqlite3.connect(f"file:{ROOT/'shared_rollout_db/rollout.sqlite'}?mode=ro",uri=True)
    outcome=defaultdict(lambda:np.full((16,3),-1,np.int8))
    for c in cases:
        row=con.execute("SELECT success,collision,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?",(c['rollout_uid'],)).fetchone()
        assert row is not None and row[3]==0
        outcome[(c['state_index'],c['eta_index'])][c['future_index']]=row[:3]
    con.close()
    assert all((v>=0).all() for v in outcome.values())
    picks=frozen["choice_indices"]
    measurements={};per_state={}
    for method,choice in picks.items():
        vals=np.asarray([outcome[(i,j)] for i,j in enumerate(choice)])
        s=np.sum((vals[:,:,0]==1)&(vals[:,:,2]==0),axis=1)
        u=np.sum(vals[:,:,2]==1,axis=1)
        col=np.sum(vals[:,:,1]==1,axis=1)
        per_state[method]=dict(success=s,B15=s>=15,unknown=(s<15)&(s+u>=15),collision=col)
        measurements[method]=dict(B15_confirmed=int(np.sum(s>=15)),
                                  B15_possible=int(np.sum(s+u>=15)),
                                  mean_success_lower=float(np.mean(s/16)),
                                  mean_success_upper=float(np.mean((s+u)/16)),
                                  collisions=int(col.sum()),numerical_unknown=int(u.sum()))
    per_state_rows=[]
    ids=frozen["state_uids"]
    eta_by={(c['state_index'],c['eta_index']):c['eta'] for c in cases}
    for i in range(N):
        for method,choice in picks.items():
            j=choice[i];values=outcome[(i,j)]
            succ=int(np.sum((values[:,0]==1)&(values[:,2]==0)))
            numerical=int(np.sum(values[:,2]==1))
            per_state_rows.append(dict(state_uid=ids[i],state_index=i,method=method,
                                       eta_index=j,eta=eta_by[(i,j)],
                                       success_count=succ,numerical_unknown=numerical,
                                       collision_count=int(np.sum(values[:,1]==1)),
                                       B15_confirmed=bool(succ>=15),
                                       B15_possible=bool(succ+numerical>=15)))
    write(DEST/"per_state_selected_outcomes.json",per_state_rows)
    comparisons={}
    for other in ("within_eta_wide","within_state_shuffled","Safety","loso_state","loso_eta_wide"):
        a=per_state["within_state"];b=per_state[other]
        comparisons[f"within_state_minus_{other}"]=dict(
            B15=bootstrap(a["B15"].astype(float)-b["B15"].astype(float)),
            meanQ=bootstrap(a["success"]/16-b["success"]/16))
    for other in ("loso_eta_wide","Safety"):
        a=per_state["loso_state"];b=per_state[other]
        comparisons[f"loso_state_minus_{other}"]=dict(
            B15=bootstrap(a["B15"].astype(float)-b["B15"].astype(float)),
            meanQ=bootstrap(a["success"]/16-b["success"]/16))
    results=dict(schema="fresh_toy48_selected_only_result_v1",states=N,
                 independent_state_bootstrap_replicates=10000,
                 selected_cases_sha256=sha(DEST/"selected_cases.json"),
                 choices_sha256=sha(DEST/"choices.json"),
                 oracle_unavailable_by_design=True,
                 methods=measurements,paired=comparisons)
    write(DEST/"metrics.json",results)
    lines=["# Prospective selected-only Toy FF Q(s,η) confirmation","",
           "48 physical TEST states were fixed before critic scoring; all five selectors and Safety",
           "used identical state-specific K16 proposals and 16 fixed future seeds.",
           "Only selected η outcomes were simulated, so no oracle is estimated.","",
           "| Method | B15 confirmed/48 | B15 possible/48 | Mean success lower–upper | Collisions |",
           "|---|---:|---:|---:|---:|"]
    for method,z in measurements.items():
        lines.append(f"| {method} | {z['B15_confirmed']}/48 | {z['B15_possible']}/48 | "
                     f"{z['mean_success_lower']:.3f}–{z['mean_success_upper']:.3f} | {z['collisions']} |")
    lines += ["","| Paired comparison | B15 difference [95% state CI] | Mean Q difference [95% state CI] |",
              "|---|---:|---:|"]
    for name,z in comparisons.items():
        b=z['B15'];q=z['meanQ']
        lines.append(f"| {name} | {100*b['mean']:+.1f} [{100*b['ci95'][0]:+.1f}, {100*b['ci95'][1]:+.1f}] pp | "
                     f"{100*q['mean']:+.1f} [{100*q['ci95'][0]:+.1f}, {100*q['ci95'][1]:+.1f}] pp |")
    lines += ["","Toy's original historical target set was not reused as this TEST set.",
              "The 48 states are the first 48 of the previously frozen, globally sampled fresh Toy TEST panel.",
              "The source model checkpoints and choices were frozen before these rollout outcomes.",""]
    (DEST/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(methods={k:v['B15_confirmed'] for k,v in measurements.items()},
                          paired={k:v['B15'] for k,v in comparisons.items()})))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("freeze","plan","run","evaluate"))
    p.add_argument("--index",type=int);a=p.parse_args()
    if a.action=="freeze":freeze()
    elif a.action=="plan":plan()
    elif a.action=="evaluate":evaluate()
    else:
        assert a.index is not None
        run(a.index)
