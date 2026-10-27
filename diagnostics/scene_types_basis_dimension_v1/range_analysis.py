"""Saved-record comparison of equal-budget, paired B3 Goal-range controls."""
from core import *

def difference_interval(method,reference,seed):
    def key(r):return(r['type_id'],r['controller'],r['state_index'])
    ref={key(r):r for r in reference};groups={};observed=[]
    for r in method:
        k=key(r)
        if k not in ref:continue
        d=int(r['success'])-int(ref[k]['success']);observed.append(d);groups.setdefault(k[0],{}).setdefault(k[2],[]).append(d)
    rng=np.random.default_rng(seed);num=np.zeros(5000);den=np.zeros(5000)
    for states in groups.values():
        blocks=list(states.values());ss=np.array([sum(b)for b in blocks]);nn=np.array([len(b)for b in blocks]);ix=rng.integers(len(blocks),size=(5000,len(blocks)));num+=ss[ix].sum(1);den+=nn[ix].sum(1)
    return dict(paired_cases=len(observed),success_rate_difference=float(np.mean(observed))if observed else None,paired_physical_block_bootstrap95=np.quantile(num/np.maximum(den,1.),[.025,.975]).tolist()if observed else None)

def main(rows,stats,case_rows):
    labels=read(HERE/'GOAL_RANGE_LABELS.json');results={};selected={};points={}
    for name,ll in labels.items():
        grouped={}
        for x in ll:
            k=(x['type_id'],x['controller'],x['state_index']);grouped.setdefault(k,[]).append(x)
            if x['uid']in rows:points[(name,*k,x['candidate'])]=rows[x['uid']]
        rr=[];topup=[]
        for k,group in grouped.items():
            group.sort(key=lambda x:x['candidate']);pool=[rows[x['uid']]for x in group if x['uid']in rows]
            if pool:rr.append(next((r for r in pool if r['success']),pool[0]))
            tail=[rows[x['uid']]for x in group if x['candidate']>=16 and x['uid']in rows]
            if tail:topup.append(next((r for r in tail if r['success']),tail[0]))
        selected[name]=rr;uids={x['uid']for x in ll if x['uid']in rows}
        results[name]=dict(K=32,overall=stats(rr),by_type={str(t):stats([r for r in rr if r['type_id']==t])for t in range(1,18)},all_candidate_outcomes=stats([rows[u]for u in uids]),unique_saved_candidates=len(uids),all_pool_compute_seconds=sum(rows[u].get('wall_seconds',0.)for u in uids),scope='Supplementary equal-budget original-three-column control; common first16 are exactly the primary B3 frozen pool. Not included in the primary dimension verdict.')
        results[name]['enough_types']=sum(v['success']>=6 for v in results[name]['by_type'].values())
        results[name]['predefined_topup_only16']=dict(overall=stats(topup),enough_types=sum(sum(r['success']for r in topup if r['type_id']==t)>=6 for t in range(1,18)),scope='Descriptive fixed-index16..31 subpool, all points frozen before their outcomes. It is not a refit or a replacement primary family; the originalB3 fallback16 is excluded here.')
    paired=[]
    for key,narrow in points.items():
        if key[0]!='narrow32'or key[-1]<16:continue
        wide=points.get(('wide32',*key[1:]))
        if wide is None:continue
        assert narrow['spec']['eta'][1:]==wide['spec']['eta'][1:]
        assert not narrow['spec'].get('coefficients')and not wide['spec'].get('coefficients')
        paired.append(dict(type_id=key[1],controller=key[2],state_index=key[3],candidate=key[4],narrow_uid=narrow['rollout_uid'],wide_uid=wide['rollout_uid'],narrow_G=narrow['spec']['eta'][0],wide_G=wide['spec']['eta'][0],narrow_success=narrow['success'],wide_success=wide['success']))
    def k(r):return(r['type_id'],r['controller'],r['state_index'])
    narrow_map={k(r):r for r in selected['narrow32']};wide_map={k(r):r for r in selected['wide32']}
    rng=np.random.default_rng(2026105801);num=np.zeros(5000);den=np.zeros(5000)
    diffs={}
    for key,r in wide_map.items():
        if key not in narrow_map:continue
        diffs.setdefault(key[0],{}).setdefault(key[2],[]).append(int(r['success'])-int(narrow_map[key]['success']))
    for states in diffs.values():
        blocks=list(states.values());ss=np.array([sum(b)for b in blocks]);ns=np.array([len(b)for b in blocks]);ix=rng.integers(len(blocks),size=(5000,len(blocks)))
        num+=ss[ix].sum(1);den+=ns[ix].sum(1)
    wide_success={k(r)for r in selected['wide32']if r['success']};family_rescues={}
    for j,rr in case_rows.items():
        success={k(r)for r in rr if r['success']}
        family_rescues[str(j)]=dict(family_succeeds_wide_B3_fails=[list(x)for x in sorted(success-wide_success)],wide_B3_succeeds_family_fails=[list(x)for x in sorted(wide_success-success)],paired_comparison=difference_interval(rr,selected['wide32'],2026105802+j),scope='Primary family Oracle@16 versus more generously budgeted widened B3@32; finite pools only, no continuous-family impossibility assertion.')
    output=dict(protocol_sha256=sha(HERE/'GOAL_RANGE_FROZEN.json'),complete=read(HERE/'goal_range_holdout_execution.json')['complete'],results=results,paired_topup_points=paired,paired_point_wide_rescues=sum(p['wide_success']and not p['narrow_success']for p in paired),paired_point_narrow_rescues=sum(p['narrow_success']and not p['wide_success']for p in paired),oracle_success_difference=results['wide32']['overall']['success']-results['narrow32']['overall']['success'],paired_physical_block_rate_difference_bootstrap95=np.quantile(num/np.maximum(den,1.),[.025,.975]),family_rescues_over_widened_B3_32=family_rescues,interpretation='Pools were frozen without primary heldout outcome inspection, after primary execution had begun. This is a supplementary Goal-coordinate experiment, not a change to the preregistered seven-family verdict. Matched per-point changes vary only G; pool coverage remains finite candidate evidence.')
    dump(HERE/'GOAL_RANGE_ABLATION_RESULTS.json',output);return results
