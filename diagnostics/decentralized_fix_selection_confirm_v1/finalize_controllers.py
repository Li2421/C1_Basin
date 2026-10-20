from support import *
G=guard()
def main():
 prereg();p=read(HERE/'CONTROLLERS_PENDING_GAUSSIAN.json');base=read(D/'controller_universal_eta_schedule_v1/protocol.json')['base_profiles'];files=[]
 for r in read(HERE/'GAUSSIAN_CONFIGS.json'):
  folder=HERE/'gaussian'/r['name'];receipt=read(folder/'COMPLETE.receipt.json')
  for path,h in receipt['files'].items():assert sha(path)==h,path
  v=dict(base[r['recipe']]['payload'],controller_class='G',flow_checkpoint_path=str(folder/'best.msgpack'),flow_checkpoint_sha256=sha(folder/'best.msgpack'),nominal_parameters=[],Gaussian_training_recipe=r['recipe'],Gaussian_seed=r['seed']);p['profiles'][r['name']]=dict(payload=v);files.extend([folder/'COMPLETE.receipt.json',folder/'best.msgpack'])
 assert len(p['controller_order'])==58 and len(p['profiles'])==58;p['eta']=read(D/'decentralized_pipeline_v1/main_protocol.json')['eta'];write(HERE/'CONTROLLERS.json',p);seal('ALL_CONTROLLERS_FROZEN.receipt.json',[HERE/'CONTROLLERS.json',HERE/'FROZEN_EXISTING_CHECKPOINTS.json']+files,before_new_instance_diagnostics=True,existing_retrained=0,Gaussian_newly_trained=4,guard=guard());source_inventory('imports_controllers_frozen.json');print('58 controllers frozen',flush=True)
if __name__=='__main__':main()
