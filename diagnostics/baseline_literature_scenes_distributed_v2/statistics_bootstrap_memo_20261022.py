"""Sealed CLI alias shared by the existing continuation and batch-file stem."""
from experiment import *
from registry import verify_receipt


def main():
    verify_receipt('BOOTSTRAP_EXACT_MEMO_LAUNCHER_FIX_20261022.receipt.json')
    from statistics_driver_bootstrap_memo_20261022 import main as run
    run()


if __name__=='__main__':main()
