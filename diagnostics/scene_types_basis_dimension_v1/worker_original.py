import runner
import runtime
original=runtime.run
def run(c):
    if c['controller_payload'].get('validation_engine')=='literal_original_RN_run':return runner.RN.run(c)
    return original(c)
runtime.run=run
if __name__=='__main__':
    import sys
    runner.worker_main(sys.argv[1],int(sys.argv[2]))
