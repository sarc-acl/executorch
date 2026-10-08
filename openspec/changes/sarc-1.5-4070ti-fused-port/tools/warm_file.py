#!/usr/bin/env python3
"""warm_file.py <file>: bring a model file into the page cache by mapping it and touching every page (what
`vmtouch -t` does; vmtouch is not on this host and nothing is installed). A plain read (`cat`, `dd`) is not
enough here: with the cache full, pages that were only read through a file descriptor are the first to be
reclaimed, and the file is partly gone again before the runner maps it. No privilege, no kernel setting."""
import mmap, os, sys

with open(sys.argv[1], "rb") as f:
    size = os.fstat(f.fileno()).st_size
    with mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ) as m:
        m.madvise(mmap.MADV_WILLNEED)
        for _ in range(2):
            s = 0
            for off in range(0, size, mmap.PAGESIZE): s += m[off]
