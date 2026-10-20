"""TS5 descriptive nominal/safety teacher references, using existing labels only."""
from core import *
from audit_utils import source_inventory
OLD=D/'teacher_student_macflow_v1'
def summarize(point,draw):return dict(mean=float(point),CI95=np.quantile(draw,[.025,.975]).tolist())
def main():
    assert (HERE/'ANALYSIS_COMPLETE.receipt.json').exists()
    results=read(HERE/'results.json');raw={r['rollout_uid']:r for r in read(OLD/'teacher_DB_ROWS.json')};cases=read(OLD/'teacher_CASES.json');z=np.load(HERE/'analysis_arrays.npz');all_names=z['controller_names'].tolist();out={}
    lines=['# TS5 descriptive teacher supplementary-arm reuse','', 'No new rollouts. Matched state/seed; controller then state clustered paired bootstrap5000, seed2026103032. Teacher rows are DEVELOPMENT references. Unknowns preserved as lower0/upper1, never rerun. Endpoint status is reported in results.json; these teacher references are descriptive.','', '| Set | Arm | Teacher success lower [95% CI] | Teacher success upper | Student−teacher lower [95% CI] | Identification interval |','|---|---|---|---:|---|---|']
    for setname in ['CONFIRMATORY','SENSITIVITY','DEVELOPMENT']:
        names=results[setname]['controllers'];C=len(names)
        if not C:continue
        t=np.full((C,32,2,2),np.nan);u=np.zeros_like(t);collision=np.zeros_like(t);timeout=np.zeros_like(t)
        for c in cases:
            if c['base_controller_name'] not in names:continue
            r=raw[c['rollout_uid']];ix=(names.index(c['base_controller_name']),c['state_index'],c['seed'],['nominal only','safety baseline'].index(c['arm']));t[ix]=0 if r['numerical_failure'] else r['success'];u[ix]=r['numerical_failure'];collision[ix]=r['collision'];timeout[ix]=r['timeout']
        assert np.isfinite(t).all();iy=[all_names.index(n) for n in names];s=z['success'][iy,...,:2];su=z['unknown'][iy,...,:2];rng=np.random.default_rng(2026103032);controllers=rng.integers(0,C,(5000,C));states=rng.integers(0,32,(5000,C,32))
        def draw(x):return x.mean(2)[controllers[:,:,None],states,:].mean((1,2))
        bt=draw(t);bs=draw(s);bu=draw(u);bsu=draw(su);out[setname]={}
        for j,arm in enumerate(['nominal only','safety baseline']):
            v=dict(teacher_success=summarize(t[...,j].mean(),bt[:,j]),teacher_success_upper=summarize((t+u)[...,j].mean(),(bt+bu)[:,j]),teacher_collision=summarize(collision[...,j].mean(),draw(collision)[:,j]),teacher_timeout=summarize(timeout[...,j].mean(),draw(timeout)[:,j]),teacher_unknown=int(u[...,j].sum()),student_minus_teacher=summarize((s-t)[...,j].mean(),(bs-bt)[:,j]),identification_interval=[float((s-t-u)[...,j].mean()),float((s+su-t)[...,j].mean())],CI_unknown_envelope=[float(np.quantile((bs-bt-bu)[:,j],.025)),float(np.quantile((bs+bsu-bt)[:,j],.975))]);out[setname][arm]=v;ts=v['teacher_success'];delta=v['student_minus_teacher'];lines.append(f"| {setname} | {arm} | {ts['mean']:.2%} {ts['CI95']} | {v['teacher_success_upper']['mean']:.2%} | {delta['mean']:.2%} {delta['CI95']} | {v['identification_interval']} |")
    write(HERE/'TS5_TEACHER_SUPPLEMENT.json',dict(utc=utc(),sets=out,source_cases_sha256=sha(OLD/'teacher_CASES.json'),source_rows_sha256=sha(OLD/'teacher_DB_ROWS.json'),rollouts_generated=0,primary_results_unchanged=True));(HERE/'TS5_TEACHER_SUPPLEMENT.md').write_text('\n'.join(lines)+'\n');source_inventory('imports_teacher_supplement.json');seal('SUPPLEMENT_DELIVERY.receipt.json',[HERE/'TS5_TEACHER_SUPPLEMENT.json',HERE/'TS5_TEACHER_SUPPLEMENT.md',HERE/'teacher_supplement.py',HERE/'ANALYSIS_COMPLETE.receipt.json'],new_rollouts=0,main_results_unchanged=True)
if __name__=='__main__':main()
