"""Original numerical path with the same JSON-only adapter as the fast worker."""
import runner
from recording_fix import install
import sys

install()
if __name__ == '__main__':
    runner.worker_main(sys.argv[1], int(sys.argv[2]))
