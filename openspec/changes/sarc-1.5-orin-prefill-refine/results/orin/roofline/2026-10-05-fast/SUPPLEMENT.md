# NVIDIA Tegra Orin (nvgpu) supplementary measurements

## Load-to-use latency (pointer chasing)

One invocation follows a host-built chain `j = next[j]`; every load depends on the previous one. Each dispatch walks the whole chain (2^16..2^20 loads); latency is the differential per-load time (fixed dispatch cost removed), in ns (clocks are not pinned unless the report says so).

| working set (random, 64 B nodes) | latency ns |
|---:|---:|
| 1 KiB | 78.4 |
| 4 KiB | 78.4 |
| 16 KiB | 78.4 |
| 4096 KiB | 1049.9 |

![latency](latency.png)
