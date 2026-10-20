from capability_stats import *
test()
a=np.zeros((14,16,2,1));a[:,::2,0]=1;a[:,1::2,1]=1;r=summarize(a,a,17,(a,a));assert r['median_max_success_pct']==[50.,50.] and r['paired_difference_of_medians_pp']==[0.,0.] and r['paired_difference_95CI_pp']==[0.,0.]
print('Paired identical-radius bootstrap exactzero PASS')
