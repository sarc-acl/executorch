# Radeon RX 7900 XTX supplementary measurements

## Load-to-use latency (pointer chasing)

One invocation follows a host-built chain `j = next[j]`; every load depends on the previous one. Each dispatch walks the whole chain (2^16..2^20 loads); latency is the differential per-load time (fixed dispatch cost removed), in ns (clocks are not pinned unless the report says so).

| working set (random, 64 B nodes) | latency ns |
|---:|---:|
| 16384 KiB | 155.2 |
| 16384 KiB | 155.7 |
| 65536 KiB | 173.7 |
| 65536 KiB | 174.5 |

![latency](latency.png)

## ERT (Empirical Roofline Toolkit) sweep

Each element: load 16 B, F dependent FMAs per component in ERT's Horner form (`y = y*x + a`, which a compiler cannot collapse), store 16 B. AI = F*8/32 FLOP/byte.

| F | AI | WG | GFLOP/s | GB/s |
|---:|---:|---:|---:|---:|
| 1 | 0.25 | 256 | 697.4 | 2789.5 |
| 1 | 0.25 | 256 | 698.2 | 2792.7 |
| 1 | 0.25 | 256 | 697.3 | 2789.0 |
| 2 | 0.5 | 256 | 1375.9 | 2751.7 |
| 2 | 0.5 | 256 | 1375.9 | 2751.8 |
| 2 | 0.5 | 256 | 1377.2 | 2754.5 |
| 4 | 1 | 256 | 2746.2 | 2746.2 |
| 4 | 1 | 256 | 2753.3 | 2753.3 |
| 4 | 1 | 256 | 2738.1 | 2738.1 |
| 8 | 2 | 256 | 5395.7 | 2697.9 |
| 8 | 2 | 256 | 5461.3 | 2730.6 |
| 16 | 4 | 256 | 10938.5 | 2734.6 |
| 16 | 4 | 256 | 10916.2 | 2729.1 |
| 16 | 4 | 256 | 10915.3 | 2728.8 |
| 32 | 8 | 256 | 21763.1 | 2720.4 |
| 32 | 8 | 256 | 21577.5 | 2697.2 |
| 32 | 8 | 256 | 21641.2 | 2705.1 |
| 64 | 16 | 256 | 20747.8 | 1296.7 |
| 64 | 16 | 256 | 37666.2 | 2354.1 |
| 64 | 16 | 256 | 28070.4 | 1754.4 |
| 64 | 16 | 256 | 36971.1 | 2310.7 |
| 128 | 32 | 256 | 42699.1 | 1334.3 |
| 128 | 32 | 256 | 21401.2 | 668.8 |
| 128 | 32 | 256 | 42650.8 | 1332.8 |
| 256 | 64 | 256 | 43584.6 | 681.0 |
| 256 | 64 | 256 | 43160.3 | 674.4 |
| 256 | 64 | 256 | 21510.6 | 336.1 |
| 256 | 64 | 256 | 35414.8 | 553.4 |
| 512 | 128 | 256 | 44424.2 | 347.1 |
| 512 | 128 | 256 | 35657.5 | 278.6 |
| 512 | 128 | 256 | 21680.7 | 169.4 |
| 512 | 128 | 256 | 44147.7 | 344.9 |
| 1024 | 256 | 256 | 43841.9 | 171.3 |
| 1024 | 256 | 256 | 21553.0 | 84.2 |
| 1024 | 256 | 256 | 43187.5 | 168.7 |
| 1024 | 256 | 256 | 35534.5 | 138.8 |

## Memory type (A/B)

Same kernel and 256 MiB working set; only the buffer memory type changes. Arms alternate order each repeat; medians (range).

| kernel | DEVICE_LOCAL GB/s | host-visible coherent GB/s | ratio |
|---|---:|---:|---:|
| mem_copy_v4 | 909.4 (909.4–909.4) | 908.9 (908.9–908.9) | 1.00× |
| mem_read_v4 | 1106.7 (1106.7–1106.7) | 1108.4 (1108.4–1108.4) | 1.00× |
| mem_triad_v4 | 896.4 (896.4–896.4) | 896.2 (896.2–896.2) | 1.00× |
| mem_write_v4 | 1617.0 (1617.0–1617.0) | 1608.2 (1608.2–1608.2) | 1.01× |
