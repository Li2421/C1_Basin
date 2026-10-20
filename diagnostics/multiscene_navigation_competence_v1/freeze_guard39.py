from common import *
def verify():
 pointer=read(HERE/'TRAINING_PIPELINE39_CURRENT.json');path=HERE/pointer['receipt'];assert sha(path)==pointer['sha256']
 receipt=read(path)
 for filename,expected in receipt['files'].items():assert sha(filename)==expected,('Training implementation differs from frozen receipt',filename)
 return receipt
