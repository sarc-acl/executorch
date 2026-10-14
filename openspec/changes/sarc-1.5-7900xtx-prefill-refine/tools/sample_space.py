#!/usr/bin/env python3
"""sample_space.py <survivor list> <n> <seed>: n 4w configurations drawn uniformly at random, without replacement,
from the survivors of enum_space.py (one token per line, in enumeration order: enum_space.py --list 4w). Prints the
tokens in draw order, so a prefix of the output is itself a uniform sample."""
import random, sys
toks = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]
random.Random(int(sys.argv[3])).shuffle(toks)
print("\n".join(toks[:int(sys.argv[2])]))
