import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from .audit import OUT,read,write,load,evaluate,canonical_h,fit_predict,choose_alpha


def fit(x,y,lam):
    mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    z=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    target=np.eye(3)[y]
    def fun(flat):
        w=flat.reshape(z.shape[1],3);logit=z@w
        loss=(logsumexp(logit,axis=1)-logit[np.arange(len(y)),y]).mean()+lam/2*(w[:-1]**2).sum()
        grad=z.T@(softmax(logit,axis=1)-target)/len(y)
        grad[:-1]+=lam*w[:-1]
        return loss,grad.ravel()
    result=minimize(fun,np.zeros(z.shape[1]*3),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    assert result.success,result.message
    return result.x.reshape(z.shape[1],3),mean,scale


def predict(model,x):
    w,m,s=model
    return softmax(np.column_stack(((x-m)/s,np.ones(len(x))))@w,axis=1)


def main():
    d,x,rows,states,indices,tr,va,seeds,s,n,s4,n4=load()
    protocol=read(OUT/'controller_decode_protocol.json')
    cx=d['context'][:,indices]
    # Nominal context must be exactly independent of candidate eta.
    np.testing.assert_allclose(cx[...,:10],np.broadcast_to(cx[:,:,:1,:10],cx[...,:10].shape),atol=1e-6)
    cc=cx[:,:,0,:10]
    y=np.repeat(np.arange(3),len(tr))
    curves=[]
    for lam in protocol['lambda']:
        losses=[]
        for fold in range(6):
            hold=tr[np.arange(len(tr))%6==fold];train=tr[np.arange(len(tr))%6!=fold]
            model=fit(cc[:,train].reshape(-1,10),np.repeat(np.arange(3),len(train)),lam)
            pp=predict(model,cc[:,hold].reshape(-1,10));yy=np.repeat(np.arange(3),len(hold))
            losses.append(-np.log(pp[np.arange(len(yy)),yy]).mean())
        curves.append(float(np.mean(losses)))
    lam=protocol['lambda'][int(np.argmin(curves))]
    model=fit(cc[:,tr].reshape(-1,10),y,lam)
    pc=predict(model,cc[:,va].reshape(-1,10)).reshape(3,len(va),3)
    prior=(s4[:,tr].sum(1)+.5)/(n4[:,tr].sum(1)+1.)
    p=pc@prior
    wrong=np.roll(pc,1,axis=0)@prior
    result={'protocol':protocol,'TRAIN_CV_NLL_by_lambda':curves,'chosen_lambda':lam,
        'VAL_controller_classification_accuracy':float((pc.argmax(-1)==np.arange(3)[:,None]).mean()),
        'VAL_controller_posteriors':pc.tolist(),
        'correct_context':evaluate(p,s[:,va],n[:,va],va),
        'wrong_context':evaluate(wrong,s[:,va],n[:,va],va),
        'oracle_controller_identity':evaluate(np.broadcast_to(prior[:,None,:],p.shape),s[:,va],n[:,va],va),
        'new_rollouts':0,'target_labels_read':False}
    write('controller_decode.json',result)
    # Remove only the stochastic, uncommitted reference action, not the state.
    reduced={k:v.copy() for k,v in x.items()};reduced['agents'][...,3:7]=0
    h=canonical_h(reduced);hh=np.broadcast_to(h[None,:,None,:],(3,30,16,h.shape[1]));blocks=[hh]
    alphas=read(OUT/'protocol.json')['kernel_probes']['alpha']
    alpha,cv=choose_alpha(blocks,s4,n4,tr,alphas)
    prediction=fit_predict(blocks,s4,n4,tr,va,alpha)
    metrics=evaluate(prediction,s[:,va],n[:,va],va)
    rng=np.random.default_rng(2026100307);null=[]
    for b in range(99):
        order=rng.permutation(tr);ps,pn=s4.copy(),n4.copy();ps[:,tr],pn[:,tr]=s4[:,order],n4[:,order]
        aa,_=choose_alpha(blocks,ps,pn,tr,alphas)
        pp=fit_predict(blocks,ps,pn,tr,va,aa)
        null.append(evaluate(pp,s[:,va],n[:,va],va)['NLL'])
    write('uncommitted_flow_ablation.json',{'alpha':alpha,'TRAIN_CV_NLL':cv,'metrics':metrics,
        'state_block_permutation_NLL_median':float(np.median(null)),
        'state_block_permutation_empirical_p':float((1+np.sum(np.array(null)<=metrics['NLL']))/100),
        'new_rollouts':0,'target_labels_read':False})
    print(result['VAL_controller_classification_accuracy'],result['correct_context'],flush=True)


if __name__=='__main__':main()
