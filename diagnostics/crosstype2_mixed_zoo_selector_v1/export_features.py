"""Readable export of the sealed mechanism-feature table, without refitting."""
from core import *
import csv


def main():
    if (HERE/'FEATURE_TABLE.receipt.json').exists():
        for path,digest in read(HERE/'FEATURE_TABLE.receipt.json')['files'].items():assert sha(path)==digest
        return
    for path,digest in read(HERE/'CONTROLLER_FEATURES.receipt.json')['files'].items():assert sha(path)==digest
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:
        names=z['names'];types=z['types'];features=z['features'];columns=z['feature_names'].tolist()
    target=HERE/'CONTROLLER_FEATURES.csv'
    with target.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['controller','type',*columns])
        for name,kind,row in zip(names,types,features):writer.writerow([name,kind,*[float(v) if np.isfinite(v) else '' for v in row]])
    seal('FEATURE_TABLE.receipt.json',[target,HERE/'export_features.py',HERE/'CONTROLLER_FEATURES.receipt.json'],blank_values_are_missing=True,no_new_rollouts=True)
    print('Exported sealed controller features:',len(names),'rows and',len(columns),'features')


if __name__=='__main__':main()
