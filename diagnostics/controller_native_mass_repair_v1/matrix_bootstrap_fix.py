"""Correct old matrix analysis: donor and recipient physical states both resample."""
def correct_donor_bootstrap(code):
    start=code.index('                        ml,mu,ql,qu=np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)')
    end=code.index('                        mechanisms.append',start)
    replacement='''                        ml,mu,ql,qu=(np.zeros((n,n)) for _ in range(4))
                        rate=np.zeros(n)
                        for donor in range(n):
                            wrong=np.broadcast_to(chosen[donor],chosen.shape)
                            a,b=selection_difference(lo,hi,chosen,wrong)
                            ml[:,donor]=a.mean(1);mu[:,donor]=b.mean(1)
                            a,b=mse_gain(q[donor],q,lo,hi)
                            ql[:,donor]=a.mean((1,2));qu[:,donor]=b.mean((1,2))
                            rate+=(wrong!=chosen).mean(1)/(n-1)
                        perm_selected=chosen[permutations];saved[name+'__mismatch_selected']=perm_selected
                        for label,a,b,suffix in [('matched_minus_same_scene_wrong_state',ml,mu,'__mismatch'),('wrong_state_Q_MSE_minus_matched',ql,qu,'__mismatch_Q')]:
                            pl,al,okl=donor_bootstrap(a,boot);pu,au,oku=donor_bootstrap(b,boot);good=okl&oku
                            arrays[name+suffix+'_lower']=al;arrays[name+suffix+'_upper']=au
                            comparisons.append(dict(**meta,condition=c,comparison=label,
                                identification_bounds=[pl,pu],censor_CI95=[float(np.quantile(al[good],.025)),float(np.quantile(au[good],.975))]))
'''
    return 'from diagnostics.controller_q_diversity_v1.evaluate import donor_bootstrap\n'+code[:start]+replacement+code[end:]
