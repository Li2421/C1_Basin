"""Audit reusable first actions before deciding the next learning diagnostic."""
from pathlib import Path
import json
import numpy as np
from shared_rollout_db.src.rollout_db import connect
from diagnostics.controller_crossed_coverage_v1.study import read,write,freeze,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=ROOT/'controller_crossed_coverage_loso_v1/evaluation_data'


def run():
    d=dict(np.load(BASE/'dataset.npz'));r=dict(np.load(BASE/'raw_labels.npz'))
    ids=np.flatnonzero(d['kind']=='goal');wanted={str(r['rollout_uid'][i,j,k]):(a,i,j,k) for a,i in enumerate(ids) for j in (0,5,10,15) for k in range(4) if r['observed'][i,j,k]}
    groups={};index=[]
    with connect(True) as db:
        for uid,where in wanted.items():
            x=db.execute('SELECT state_uid,eta_uid,controller_uid,seed_key,original_source_file,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
            assert x is not None and not x['conflict_quarantined']
            groups.setdefault(x['original_source_file'],{})[uid]=dict(x)
    values=np.full((len(ids),16,4,4,2),np.nan);known=np.zeros((len(ids),16,4),bool);seen=set()
    for filename,required in groups.items():
        f=Path(filename);assert f.exists()
        records=[json.loads(line) for line in f.read_text().splitlines()] if f.suffix=='.jsonl' else [read(f)]
        for x in records:
            uid=x.get('rollout_uid',x.get('case',{}).get('rollout_uid'))
            if uid not in required:continue
            assert uid not in seen;seen.add(uid);a,i,j,k=wanted[uid];row=required[uid]
            assert all(x[key]==row[key] for key in ('state_uid','eta_uid','controller_uid','seed_key'))
            assert x['success']==r['success'][i,j,k] and x['numerical_failure']==r['unknown'][i,j,k]
            act=x.get('first_action');n=int(d['entity_agent_mask'][i].sum())
            if act is not None:
                act=np.asarray(act,float);assert act.shape==(n,2) and np.isfinite(act).all();values[a,j,k,:n]=act;known[a,j,k]=True
            else:assert x['numerical_failure'],'Missing firstactionwithoutnumericalfailure'
        index.append(dict(path=filename,sha256=sha(f),used=len(required)))
    assert seen==set(wanted)
    obs=r['observed'][ids].astype(bool);num=r['unknown'][ids].astype(bool);assert (known|(num)|(~obs)).all()
    out=HERE/'cached_first_actions.npz';assert not out.exists()
    np.savez_compressed(out,first_action=values,first_action_known=known,original_matrix_row=ids,
        state_uid=d['state_uid'][ids],scene=d['scene'][ids],controller=d['controller'][ids],split=d['split'][ids],
        **{k:r[k][ids] for k in ('success','unknown','collision','deadlock','timeout','observed','rollout_uid')})
    freeze(HERE/'cache_audit.json',dict(scope='Read-onlycacheinspection;noexperimentmodelselectionortrainingyet.',
        history='EarlierFFaction/encounter experiments suppliedtheseasQinputs;they didnottrainthesamecritic tolearnactualfirst-actionprogressversuslonghorizonreturn. Oldexactinitializerinputs alreadytestedtwo-sceneanalytic6family;mustnotrepeatthatclaim asnew.',
        current_question='Potentialnexttarget:locatefailurebetween localS/C/eta executionmapping and longhorizonQ,using identicalcachedtrials and a knownsharedgoal-reference program. Designnotyetfrozen.',
        raw_outcomes=len(wanted),known_first_actions=int(known.sum()),missing_first_actions=int((obs&~known).sum()),numerical_unknown=int(num.sum()),
        nonnumerical_actions_complete=True,controller_rows=len(ids),physical_states=len(set(d['state_uid'][ids])),
        source_database='shared_rollout_db/rollout.sqlite',source_files=index,data_sha256=sha(out),new_rollout=0,new_query=0,new_fit=0))
    print(dict(cached_outcomes=len(wanted),first_actions=int(known.sum()),missing=int((obs&~known).sum()),new_rollout=0),flush=True)


if __name__=='__main__':run()
