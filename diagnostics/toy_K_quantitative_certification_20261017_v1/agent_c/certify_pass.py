"""Arithmetic-only certificate; no controller imports, projection, or rollout."""
import csv, hashlib, json, math, pathlib, time
T=time.perf_counter(); root=pathlib.Path(__file__).resolve().parent
base=root.parent; a_path=base/'agent_a/geometry_contract.json'; A=json.loads(a_path.read_text())
c=A['constants_m']; R=c['R_w']; Ds=c['D_s']; V=A['runtime_config']['max_speed']; dt=A['runtime_config']['dt']; gamma=A['cbf_config']['gamma']
Zmax=c['H']+c['h']; Zmin=.38; v=.06; D=.34
quad_A=1-R/.4; quad_C=.2*(.2-R)
wall_v=math.sqrt(4*quad_A*quad_C)
old=root.parents[1]/'tt_ff_correction_realization_20261017_v1'; rows=list(csv.DictReader((old/'tt_ff_action_validation.csv').open()))
def inside(r):
 p=json.loads(r['positions']); (x1,y1),(x2,y2)=p
 return -.65<=x1<=x2+D and -.025<=x2<=.025 and -.02<=y1<=0 and .38<=y2<=.46
selected=[r for r in rows if r['parameter']=='witness' and r['chain']=='FF']
res={
 'labels':{'reference_pass':'PROVEN','fixed_eta_TT_FF_full_domain':'UNKNOWN','historical_local_pass':'EMPIRICALLY OBSERVED'},
 'inputs':{'geometry_contract':str(a_path),'sha256':hashlib.sha256(a_path.read_bytes()).hexdigest(),'source_hashes':A['source_hashes']},
 'constants':{'R_w_m':R,'D_s_m':Ds,'gamma_per_s':gamma,'V_m_per_s':V,'dt_s':dt,'Z_max_geometric_m':Zmax,'D_exit_choice_m':D},
 'domain':{'x2':[-.025,.025],'y1':[-.02,0],'y2':[.38,.46],'x1_lower':-.65,'x1_upper':'x2 + .34','s':[-.675,.34],'z':[.38,.48]},
 'bounds':{'min_wall_h_m':.175-R,'min_pair_distance_m':Zmin,'min_pair_h_m2':Zmin**2-Ds**2,'max_pair_speed_at_domain_Zmin':gamma*math.sqrt(Zmin**2-Ds**2),'max_pair_speed_at_geometric_Zmax':gamma*math.sqrt(Zmax**2-Ds**2),'wall_endpoint_speed_sufficient_bound':wall_v,'wall_endpoint_quadratic_min_for_v06_m2':quad_C-v*v/(4*quad_A),'right_end_max_v_sufficient':1.3-.365-R,'reference_speed_m_per_s':v,'speed_ball_slack':V-v,'pass_time_upper_s':(.675+D)/v,'pass_steps_upper':math.ceil((.675+D)/(v*dt)),'4D_volume_m4':.05*.02*.08*.99},
 'Zreq_table':[{'v_m_per_s':x,'Z_req_m':math.sqrt(Ds**2+(x/gamma)**2),'within_geometric_Zmax':math.sqrt(Ds**2+(x/gamma)**2)<=Zmax,'within_certified_domain_and_wall_bound':x<=min(wall_v,gamma*math.sqrt(Zmin**2-Ds**2),V)} for x in [.025,.05,.06,.1,.2,.3,.4,.5]],
 'old_action_rows':len(rows),'old_witness_FF_records':len(selected),'old_witness_FF_records_in_certified_pass_domain':sum(inside(r) for r in selected),
 'old_t215_counterexample':[{'state_id':r['state_id'],'chain':r['chain'],'positions':json.loads(r['positions']),'eta':json.loads(r['eta']),'raw_gap':float(r['raw_delta1x_minus_delta2x']),'executed_gap':float(r['u1x_minus_u2x']),'inside_certified_domain':inside(r)} for r in rows if ':t215' in r['state_id'] and r['parameter']=='witness'],
 'new_rollouts':0,'new_projection_calls':0,'new_flow_calls':0,'training':0,
 'evidence_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'tt_ff_action_validation.csv',old/'trajectory_mechanism_validation.md',old/'trajectory_metrics.json']}}
res['basis_witness_uniform_bounds']={'delta_gap_lower_m_per_s':(.5*.725/math.hypot(.725,.02)+.5*1.065/math.hypot(1.065,.46))/2-.5*.675/math.hypot(.675,.38), 'delta2y_lower_m_per_s':(.5*.38/math.hypot(.675,.38)-.5*.46/math.hypot(1.065,.46))/2, 'eta':[.5,0,.5], 'status':'PROVEN analytic bounds with separate saturated/unsaturated relative-vector branches'}
res['wall_seconds']=time.perf_counter()-T
(root/'results.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'bounds':res['bounds'],'Zreq_table':res['Zreq_table'],'matching_old_records':res['old_witness_FF_records_in_certified_pass_domain'],'wall_seconds':res['wall_seconds']},indent=2))
