from common import *
from pipeline import call

def main():
 assert (HERE/'A_COMPLETE.receipt.json').exists()
 if (HERE/'PSTAR_DECISION.receipt.json').exists() and read(HERE/'PSTAR_DECISION.receipt.json')['authorized_rollouts']:assert (HERE/'B_COMPLETE.receipt.json').exists()
 call('plot_curves.py');call('immutability.py');call('audit.py','END');call('delivery.py');call('review.py')
if __name__=='__main__':
 try:main()
 except Exception as ex:write(HERE/'FINALIZATION_ERROR.json',dict(utc=utc(),error=repr(ex)));raise
