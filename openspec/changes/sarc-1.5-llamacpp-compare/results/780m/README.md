# Radeon 780M: results and one deviation from the validity rule

Session `s1`, 2026-10-06, 14 arms, taken while the tuning campaign was held. `cells-strict.csv` applies the
campaign's validity rule as it stands; `cells.csv` additionally counts runs whose only finding is `clock_low`.

## Why `clock_low` runs are counted on this device

The campaign's rule rejects a run whose median GPU clock in the prefill window is below 2700 MHz. It was set
for the campaign's own arms, which all reach 2700 to 2800 MHz. In this session the clock turned out to be a
property of the workload, not a disturbance:

- every run of an arm lands on the same clock and the same speed. Stock `4w`: 2606, 2509 and 2421 MHz on 1B, 3B
  and 8B in all seven runs each, spread of tok/s 0.9 to 1.6 %. llama.cpp Q4_0 on 8B: 2623 to 2644 MHz in all
  fourteen runs, spread 0.4 to 0.7 %.
- the arms that fall below the floor are the ones with the longest prefill (7 to 11 s) and are the same ones
  in every repetition, whatever ran before them; arms above the floor stay above it.

The chip limits its clock by power and temperature (the package reached 87 C in the 8B part), and a workload
that draws more or runs longer settles lower. Rejecting those runs would remove whole arms (stock `4w`, the
SARC 8B `4w` reference, llama.cpp Q4_0 on 3B and 8B) rather than outliers. They are therefore reported, with
their clock (`clk_med_mhz`) and the number of runs this applies to (`n_accepted`), and marked wherever they are
quoted. No run with any other finding is counted.

What this means for reading the numbers: an arm that ran at a lower clock was measured at the speed this
device gives that workload in a session of this length, which is the quantity of interest, but it is not the
speed at 2700 MHz. The SARC 8B `4w` arm reads 507 tok/s here at 2633 MHz against 526 in the September table.

## Result (tok/s, median; llama.cpp = the higher of its two timers at its better setting)

| model | stock 4w | stock 8da4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Q4_0 | llama.cpp Q4_K_M | tuned 4w / llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 1176* | 1620 | 2681 | 2538 | 3828 | 3690 | 2880 | 2081 | 1.33 |
| 3B | 404* | 560 | 1126 | 1044 | 1450 | 1365 | 959 | 690 | 1.51 |
| 8B | 184* | 275* | 507* | 483 | 630 | 610 | 451* | 308 | 1.40 |

\* counted under the deviation above. llama.cpp `default` and `best` are within 2 % of each other on this device
and its two timers within 9 % (llama-bench higher on 1B, llama-completion higher on 8B). Q4_K_M is 28 to 32 %
slower than Q4_0 here, unlike on the Arc B580 where the two were level.

Text check: every arm continues the real-text prompt fluently; ExecuTorch arms agree with each other except
8B `8da4w` (known); llama.cpp gives the same next word as ExecuTorch on 3B and a different, equally plausible one on 1B and 8B.
