"""Draft exact G0–G5 recipes without training or reading held-out outcomes."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    source=ROOT/'diagnostics/axis2_controller_v1/GAUSSIAN_CONFIGS.json'
    old=json.loads(source.read_text());rows=[]
    for family in ('chunking','recurrent','cvae','implicit'):
        for i,c in enumerate(old):
            cfg=dict(c)
            cfg.update(name=f'{family}{i}',family=family,source_instance=c['name'],
                       seed=c['seed'],locked=family=='implicit',
                       checkpoint_selection='minimum fixed dev loss, earliest tie',
                       source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            if family=='chunking':
                cfg.update(chunk_length=8,replan_interval=4,temporal_ensemble=False,
                           loss='masked MSE; pad final action, never cross episode')
            elif family=='recurrent':
                cfg.update(gru_hidden=256,head_hidden=[256,256],tbptt=32,parallel_streams=8,
                           loss='Gaussian NLL; 8 sequences x32=256 token batch',
                           state_rule='carry detached state across consecutive windows; reset at episode boundaries')
            elif family=='cvae':
                cfg.update(latent_dim=4,kl_weight=.5,encoder_hidden=[256]*3,decoder_hidden=[256]*3,
                           loss='MSE plus 0.5*latent KL')
            else:
                cfg.update(negative_samples=256,negative_distribution='uniform physical raw action [-1,1]^8, then same normalization',
                           inference_iterations=3,inference_samples=1024,elite_count=103,loss='InfoNCE')
            rows.append(cfg)
    path=HERE/'TRAINING_CONFIGS_DRAFT.json'
    assert not path.exists()
    path.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
