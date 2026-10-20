from common import *
from data import load_labels
from statistics import source_choices

def main():
 cap=read(HERE/'CAPABILITY_GATE.json');n=cap['source_N']
 if n is None:out=dict(utc=utc(),source_N=None,status='GAP',reason='No capability-qualified source',choices=None)
 else:
  source=load_labels('source',n);out=dict(utc=utc(),source_N=n,status='FROZEN',classes=source['classes'],choices=source_choices(source));np.savez_compressed(HERE/'SOURCE_LABELS_FROZEN.npz',lower=source['lower'],upper=source['upper'],classes=np.asarray(source['classes']))
 write(HERE/'TRANSFER_CHOICES.json',out);paths=[HERE/'TRANSFER_CHOICES.json',HERE/'statistics.py']+([HERE/'SOURCE_LABELS_FROZEN.npz'] if n is not None else []);seal('TRANSFER_CHOICES_FROZEN.receipt.json',paths,before_target_main=True,target_eval_labels_read=False);print('Transfer choices frozen',n,flush=True)
if __name__=='__main__':main()
