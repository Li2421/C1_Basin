"""Ensemble generator trained on EXACTLY the original data object (critic_data_with_test.pkl, as step3), then proposals for
target scenes whose features come from critic_data_with_test2.pkl.  usage: gen_ens_t2.py TARGETS.json"""
import json,pickle,sys
import gen_ens as GE
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data_with_test.pkl','rb'))
nm,models=GE.train_ens(data,sp['train'])
data2=pickle.load(open('critic_data_with_test2.pkl','rb'));tg=json.load(open(sys.argv[1]))
for v in tg:
 if v not in data:data[v]=data2[v]
print('PROPS',json.dumps({v:GE.props(data,nm,models,v,'hetset-ens').tolist() for v in tg}))
