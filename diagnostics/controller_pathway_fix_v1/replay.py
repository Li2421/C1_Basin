from util import *
import time
def main(i):
    import jax
    jax.config.update('jax_enable_x64',False)
    t=read(HERE/'REPLAY_TASKS.json')[i];arm={'B0':'R0','B1':'R3','B2':'R_oracle'}[t['arm']]
    out=HERE/'replay'/task_name(t);out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():return
    start=__import__('time').time()
    if t['bed']=='blend':
        p=read(R5/'stage_2/protocol.json');f=OldFeatures(p,arm);states=p['states'][96:];folder=R5/'stage_2/fits'/f"LOPO_fold{t['fold']}_{t['coverage']}_{arm}"
        norm=np.load(folder/'normalization.npz');X=np.array([(f.inputs(n,states)-norm['mean'])/norm['scale'] for n in t['held']]);saved=np.load(folder/'predictions.npz')['held']
        pred=frozen_predict(folder,X.reshape(-1,536)).reshape(saved.shape)
        outcomes=np.load(R5/'stage_2/TEST_ENDPOINT_LABELS.npz')['outcomes'];names=p['controller_order'];historical=read(R5/'stage_2/results.json')['rows']
        key='success_lower'
    else:
        folder=R4/'fits'/f"fold{t['fold']}_{arm}";norm=np.load(folder/'normalization.npz');_,_,_,X,_,_=old4_build(t['fold'],arm)
        saved=np.load(folder/'predictions.npz')['held'];pred=frozen_predict(folder,(X-norm['mean'])/norm['scale']);pred=pred[:,None];saved=saved[:,None]
        outcomes=np.load(R4/'LABELS.npz')['outcomes'][:,128:,:,4:8];names=read(R4/'protocol.json')['controller_order'];historical=read(R4/'results.json')['rows'];key='selected_success_lower'
    error=float(np.max(np.abs(pred-saved)));same=np.array_equal(pred.argmax(-1),saved.argmax(-1));success=[]
    for j,n in enumerate(t['held']):
        Y=(outcomes[names.index(n)]==1).mean(-1);choice=pred[:,j].argmax(-1);rate=float(np.take_along_axis(Y[None],choice[...,None],axis=-1).mean())
        r=next(r for r in historical if r['held_controller']==n and r['arm']==arm and (t['bed']=='real' or (r['split']=='LOPO' and r['coverage']==t['coverage'])))
        assert abs(rate-r[key])<1e-10,(n,rate,r[key]);success.append(rate)
    receipt=dict(status='PASS' if error<=2e-5 and same else 'FAIL',task=t,max_logit_abs_error=error,argmax_identical=same,success_identical=True,success_lower=success,elapsed_seconds=time.time()-start,jax_enable_x64=False,checkpoint_sha256=[sha(p) for p in read(folder/'complete.json')['checkpoints']],historical_folder=str(folder),devices=[str(d) for d in jax.devices()])
    np.savez_compressed(out/'predictions.npz',held=pred);write(out/'receipt.json',receipt);assert receipt['status']=='PASS',receipt
    print(json.dumps(receipt))
if __name__=='__main__':main(int(sys.argv[1]))
