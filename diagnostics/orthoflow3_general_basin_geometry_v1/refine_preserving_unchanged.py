#!/usr/bin/env python3
from audit import *
from families import run
if __name__=='__main__':
 run(sys.argv[1],True,kinds=['minimal_polyhedral_support_quadratic_exclusion'],reuse_tag=sys.argv[2])
