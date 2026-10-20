from prefix import *
def c(H,arm,t=500):
 a=[1.,2.,3.];b=[4.,5.,6.];return dict(H=H,arm=arm,eta_a=a,eta_b=b if arm.startswith('S') else a,switch_step=t,progress_threshold=.5)
g=np.ones((2,2))*10;pos=np.zeros((701,2,2));old=c(700,'F13',700)
assert choose_prefix(c(500,'F13',500),old,pos,g,'timeout')==dict(steps=500,kind='BUDGET_PREFIX',termination='timeout',first_action_difference=None)
assert choose_prefix(c(1000,'F13',1000),old,pos,g,'timeout')['steps']==700
assert choose_prefix(c(500,'Srel',357),old,pos,g,'timeout')['steps']==357
assert choose_prefix(c(1000,'F13',1000),old,pos[:301],g,'numerical_failure')['kind']=='UNKNOWN_REUSE'
assert choose_prefix(c(1000,'Srel',300),old,pos[:301],g,'numerical_failure')['kind']=='CONTINUE'
assert choose_prefix(c(1000,'Srel',300),old,pos[:301],g,'success')['kind']=='TERMINAL_REUSE'
assert choose_prefix(c(1000,'F13',1000),old,pos[:301],g,'success')['kind']=='TERMINAL_REUSE'
assert choose_prefix(c(500,'F13',500),old,pos,g,'numerical_failure')['kind']=='BUDGET_PREFIX'
assert choose_prefix(c(1000,'Sabs',500),c(700,'S500',500),pos,g,'timeout')['steps']==700
assert choose_prefix(c(1000,'Sprog'),old,pos,g,'timeout')['steps']==700
print('Prefix causal compatibility tests PASS')
