#!/usr/bin/env python3
"""rgp_chunks.py <capture.rgp> [--elf <dir>]: what can be read from a Radeon GPU Profiler capture without the viewer.

Walks the chunk list of the file (layout as written by Mesa's ac_rgp.c) and prints every chunk with its size, then
what the simple chunks hold: the ASIC description, the queue event timings (GPU time of each submitted command
buffer), the performance-counter database (SPM: one row per counter with the sum over the samples, for the cache
counters RADV programs: TCP, GL1C and GL2C requests and misses) and the code-object database (one ELF per pipeline;
--elf writes them out, llvm-objdump --triple=amdgcn can disassemble their .text). The thread trace itself (SQTT
data: wavefront and per-instruction timing tokens) is only sized here; it needs the viewer."""
import struct, sys, os
f = open(sys.argv[1], "rb").read(); elfdir = sys.argv[sys.argv.index("--elf") + 1] if "--elf" in sys.argv else None
magic, vmaj, vmin, flags, chunk_off = struct.unpack_from("<IIIIi", f, 0)
print(f"file {os.path.basename(sys.argv[1])} bytes {len(f)} magic {magic:#x} version {vmaj}.{vmin} first chunk at {chunk_off}")
NAMES = {0: "ASIC_INFO", 1: "SQTT_DESC", 2: "SQTT_DATA", 3: "API_INFO", 4: "RESERVED", 5: "QUEUE_EVENT_TIMINGS", 6: "CLOCK_CALIBRATION",
         7: "CPU_INFO", 8: "SPM_DB", 9: "CODE_OBJECT_DATABASE", 10: "CODE_OBJECT_LOADER_EVENTS", 11: "PSO_CORRELATION", 12: "INSTRUMENTATION_TABLE"}
off = chunk_off; sqtt = 0; n = 0
while off + 16 <= len(f):
    tid, vminor, vmajor, size, pad = struct.unpack_from("<IHHii", f, off)
    ctype, index = tid & 0xFF, (tid >> 8) & 0xFF
    if size <= 0: break
    name = NAMES.get(ctype, f"type{ctype}"); n += 1
    extra = ""
    body = off + 16
    if ctype == 2:
        sqtt += size
    elif ctype == 0:
        # sqtt_file_chunk_asic_info: flags u64, trace_shader_core_clock u64, memory_clock u64, device_id i32, revision i32, vendor i32, ...
        # ... device_id i32, device_revision_id i32, vgprs_per_simd i32, sgprs_per_simd i32, shader_engines i32,
        # compute_unit_per_shader_engine i32, simd_per_compute_unit i32, wavefronts_per_simd i32
        fl, core, mem, dev, rev, vg, sg, se, cu, simd, waves = struct.unpack_from("<QQQiiiiiiii", f, body)
        extra = (f" shader core clock {core / 1e6:.0f} MHz, memory clock {mem / 1e6:.0f} MHz, device {dev:#x}, VGPRs per SIMD {vg}, "
                 f"shader engines {se}, compute units per engine {cu}, SIMDs per compute unit {simd}, wavefronts per SIMD {waves}")
    elif ctype == 5:
        # sqtt_file_chunk_queue_event_timings: queue_info_table_record_count u32, queue_info_table_size u32,
        # queue_event_table_record_count u32, queue_event_table_size u32
        qi, qis, qe, qes = struct.unpack_from("<IIII", f, body)
        extra = f" queues {qi}, events {qe}"
        ev = body + 16 + qis; rows = []
        for i in range(qe):
            # sqtt_queue_event_record: event_type u32, sqtt_cb_id u32, frame_index u64, queue_info_index u32, submit_sub_index u32, api_id u64, cpu_timestamp u64, gpu_timestamps[2] u64
            et, cb, frame, qidx, sub, api, cpu, g0, g1 = struct.unpack_from("<IIQIIQQQQ", f, ev + i * 56)
            rows.append((et, cb, cpu, g0, g1))
        extra += "; gpu ticks per event (type:ticks) " + " ".join(f"{et}:{g1 - g0}" for et, cb, cpu, g0, g1 in rows)
    elif ctype == 8:
        # sqtt_file_chunk_spm_db: flags u32, preamble_size u32, num_timestamps u32, num_spm_counter_info u32, spm_counter_info_size u32, sample_interval u32
        fl, pre, nts, ncnt, cisz, interval = struct.unpack_from("<IIIIII", f, body)
        extra = f" samples {nts}, counters {ncnt}, sample interval {interval}"
        ts_off = body + 24; ci_off = ts_off + 8 * nts
        print(f"chunk {n:2d} {name:26s} index {index} size {size}{extra}")
        tot = {}
        for c in range(ncnt):
            # sqtt_spm_counter_info: block u32, instance u32, event_index u32, data_offset u32 (from the chunk start), data_size u32
            block, inst, event, doff, dsz = struct.unpack_from("<IIIII", f, ci_off + c * cisz)
            vals = struct.unpack_from("<" + ("I" if dsz == 4 else "H") * nts, f, off + doff) if nts else ()
            k = (block, event); tot.setdefault(k, [0, 0]); tot[k][0] += sum(vals); tot[k][1] += 1
        for (block, event), (v, inst) in sorted(tot.items()):
            print(f"      block {block:3d} event {event:#06x}: sum over {inst} instances and {nts} samples = {v}")
        off += size; continue
    elif ctype == 9:
        # sqtt_file_chunk_code_object_database: offset u32, flags u32, size u32, record_count u32; records: size u32 + ELF
        o2, fl, sz, rc = struct.unpack_from("<IIII", f, body)
        extra = f" records {rc}"; p = body + 16
        for r in range(rc):
            rs, = struct.unpack_from("<I", f, p); blob = f[p + 4:p + 4 + rs]
            if elfdir:
                os.makedirs(elfdir, exist_ok=True); open(os.path.join(elfdir, f"code-object-{r}.elf"), "wb").write(blob)
            extra += f"; elf {r}: {rs} bytes"
            p += 4 + rs
    print(f"chunk {n:2d} {name:26s} index {index} size {size}{extra}")
    off += size
print(f"thread trace (SQTT data) bytes: {sqtt}")
