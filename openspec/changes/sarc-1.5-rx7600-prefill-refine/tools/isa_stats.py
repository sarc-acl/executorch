#!/usr/bin/env python3
"""isa_stats.py <RADV_DEBUG=shaders dump (stderr)>: per compute shader of the dump that contains v_wmma: workgroup size, shared bytes,
VGPRs used after register allocation (highest register index + 1), spill stores / loads, instruction mix of the loop blocks (ACO IR after
RA): ds_read_b64 / ds_read_b128 / ds_write_b32 / ds_write_b128, v_wmma, s_barrier, global loads. One line per shader plus the block mix of
the blocks that hold v_wmma."""
import re, sys
t = open(sys.argv[1]).read()
parts = re.split(r"(?=^shader: MESA_SHADER_COMPUTE)", t, flags=re.M)
seen = set()
for i, p in enumerate(parts):
    if "v_wmma" not in p or "After RA:" not in p: continue
    lines = p.splitlines()
    wg = re.search(r"workgroup_size: (\d+), (\d+), (\d+)", p); sh = re.search(r"shared_size: (\d+)", p)
    sect = "\n".join(lines[lines.index("After RA:"):]).split("After lowering to hw instructions:")[0]
    mx = max(int(m.group(2) or m.group(1)) for m in re.finditer(r"%\d+:v\[(\d+)(?:-(\d+))?\]", sect))
    key = (wg.group(0), sh and sh.group(1), mx, sect.count("scratch_store"), sect.count("ds_read_b128"), sect.count("ds_read_b64"))
    if key in seen: continue
    seen.add(key)
    print(f"shader#{i} wg={wg.group(1)}x{wg.group(2)}x{wg.group(3)} shared={sh and sh.group(1)} VGPRs={mx + 1} "
          f"spill_st={sect.count('scratch_store')} spill_ld={sect.count('scratch_load')} wmma={sect.count('v_wmma')} "
          f"ds_read_b128={sect.count('ds_read_b128')} ds_read_b64={sect.count('ds_read_b64')} ds_read_b32={sect.count('ds_read_b32')} "
          f"ds_write_b32={sect.count('ds_write_b32')} ds_write_b128={sect.count('ds_write_b128')} s_barrier={sect.count('s_barrier')}")
    bb, blocks = None, {}
    for l in sect.splitlines():
        m = re.match(r"^BB(\d+)$", l)
        if m: bb = int(m.group(1)); blocks[bb] = []
        elif bb is not None: blocks[bb].append(l)
    for b, ls in blocks.items():
        c = lambda s: sum(s in l for l in ls)
        if c("v_wmma"):
            print(f"   BB{b}: wmma={c('v_wmma')} ds_read_b128={c('ds_read_b128')} ds_read_b64={c('ds_read_b64')} ds_read_b32={c('ds_read_b32')} ds_write={c('ds_write')} lines={len(ls)}")
