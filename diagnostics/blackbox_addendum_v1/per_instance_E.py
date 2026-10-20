from common import *
from agent_statistics import summarize

def data(n):
 z=np.load(HERE/f'MPC_N{n}_LABELS.npz');return dict(N=n,classes=z['controller_names'].tolist(),lower=z['lower'],upper=z['upper'],events={k:z[k]for k in ['collision','timeout','unknown']})
def main():
 e=read(HERE/'E_RESULTS.json');cap=read(HERE/'MPC_CAPABILITY_GATE.json');source=data(e['source_N'])if e['source_N']else None;out={}
 for j,n in enumerate([2,10,20]):
  out[str(n)]={}
  available=data(n)if any(q['status']=='PASS'for q in cap[str(n)].values())else None
  for name in ['DM1','DM3']:
   gate=cap[str(n)][name]
   if gate['status']!='PASS':out[str(n)][name]=dict(status='GAP',reason=gate.get('reason','Frozen capability threshold not passed'),capability=gate);continue
   i=available['classes'].index(name);d=dict(N=n,classes=[name],lower=available['lower'][i:i+1],upper=available['upper'][i:i+1],events={k:v[i:i+1]for k,v in available['events'].items()});q=summarize(d,source,202610163750+j);q.update(instance=name,evidence='CONFIRMATORY',population_scope='Single fixed-parameter instance; state evidence only',source_reference_only=n==e['source_N'])
   if 'transfer_controller_minus_safety'in q['paired']:
    gain=q['paired']['transfer_controller_minus_safety']['CI_pp'];reg=q['paired']['transfer_controller_regret']['estimate_pp'];q['clean_transfer']=dict(correction_status='SUPPORTED'if gain[0]>0 else'REJECTED'if gain[1]<=0 else'INCONCLUSIVE',transfer_status='SUPPORTED'if reg[1]<=5 else'REJECTED'if reg[0]>5 else'INCONCLUSIVE',gain_CI_pp=gain,mean_regret_pp=reg)
   else:q['clean_transfer']=dict(status='GAP',reason='This exact parameter instance did not pass at source N')
   out[str(n)][name]=q
 write(HERE/'E_PER_INSTANCE_RESULTS.json',out);seal('E_PER_INSTANCE_RESULTS.receipt.json',[HERE/'E_PER_INSTANCE_RESULTS.json'],criterion_and_frozen_unit_seeds_unchanged=True,per_controller_and_pooled_as_preregistered=True,new_physical_steps=0)
if __name__=='__main__':main()
