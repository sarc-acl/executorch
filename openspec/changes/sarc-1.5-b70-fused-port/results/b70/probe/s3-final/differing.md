# Logits at the gate prompts and at every window where the candidate's top-1 differs from the parent's

## 1b-4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 1757): +0.6328; the same difference in candidate-default: +0.6562 (moved by +0.0234). KL(P || C) = 1.623e-04 nat, max |logit difference| = 0.1133.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 6.0547 | 6.0586 | 6.1133 | 6.1172 |
| 1757 | 5.4258 | 5.4258 | 5.4570 | 5.4609 |
| 1527 | 5.0977 | 5.1016 | 5.1562 | 5.1602 |

## 1b-4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 11737): +0.5938; the same difference in candidate-default: +0.5938 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 18.1406 | 18.1406 | 18.1406 | 18.1406 |
| 11737 | 17.5469 | 17.5469 | 17.5469 | 17.5469 |
| 61449 | 17.4219 | 17.4219 | 17.4219 | 17.4219 |

## 1b-4w w1280-gpl-384: parent top-1 311, candidate top-1 304 (DIFFER)

Parent-default top-2 margin (logit of id 311 minus id 304): +0.1562; the same difference in candidate-default: -0.1406 (moved by -0.2969). KL(P || C) = 7.092e-03 nat, max |logit difference| = 0.4766.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 311 | 20.0781 | 20.1562 | not run | 19.9531 |
| 304 | 20.0469 | 20.0000 | not run | 20.0938 |
| 555 | 19.5938 | 19.5938 | not run | 19.4844 |

## 1b-8da4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 1757): +0.1953; the same difference in candidate-default: +0.2578 (moved by +0.0625). KL(P || C) = 9.086e-04 nat, max |logit difference| = 0.2803.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 5.7500 | 5.8359 | 5.9375 | 5.8984 |
| 1757 | 5.5312 | 5.6406 | 5.6641 | 5.6406 |
| 1527 | 5.2969 | 5.4609 | 5.5469 | 5.4609 |

## 1b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 11737): +0.7812; the same difference in candidate-default: +0.7812 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 18.1562 | 18.1562 | 18.1562 | 18.1562 |
| 11737 | 17.3750 | 17.3750 | 17.3750 | 17.3750 |
| 40761 | 17.2500 | 17.2500 | 17.2500 | 17.2500 |

## 1b-8da4w w1536-gpl-0: parent top-1 11, candidate top-1 539 (DIFFER)

Parent-default top-2 margin (logit of id 11 minus id 539): +0.9844; the same difference in candidate-default: -0.6406 (moved by -1.6250). KL(P || C) = 2.853e-01 nat, max |logit difference| = 3.1729.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 11 | 18.9062 | 19.5938 | not run | 18.6875 |
| 539 | 19.1094 | 18.6094 | not run | 19.3281 |
| 320 | 17.0625 | 16.8750 | not run | 15.7031 |
| 4694 | 16.4375 | 16.4844 | not run | 16.3281 |

## 1b-8da4w w1280-gpl-384: parent top-1 304, candidate top-1 555 (DIFFER)

Parent-default top-2 margin (logit of id 304 minus id 311): +0.0781; the same difference in candidate-default: +0.0625 (moved by -0.0156). KL(P || C) = 1.231e-01 nat, max |logit difference| = 2.9785.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 304 | 20.0156 | 20.1719 | not run | 19.6719 |
| 311 | 19.6875 | 20.0938 | not run | 19.6094 |
| 1193 | 18.9844 | 19.5625 | not run | 19.7188 |
| 555 | 19.1250 | 19.2812 | not run | 20.1875 |

## 1b-8da4w w1024-gpl-512: parent top-1 11, candidate top-1 539 (DIFFER)

Parent-default top-2 margin (logit of id 11 minus id 539): +1.1562; the same difference in candidate-default: -0.8281 (moved by -1.9844). KL(P || C) = 3.851e-01 nat, max |logit difference| = 2.9111.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 11 | 20.1094 | 18.9688 | not run | 18.9844 |
| 539 | 18.5156 | 17.8125 | not run | 19.8125 |
| 2631 | 15.7891 | 16.6875 | not run | 15.7969 |
| 304 | 16.5938 | 16.5625 | not run | 17.0312 |

## 3b-4w gate-the2048: parent top-1 14924, candidate top-1 14924 (SAME)

Parent-default top-2 margin (logit of id 14924 minus id 16309): +0.7148; the same difference in candidate-default: +0.7109 (moved by -0.0039). KL(P || C) = 2.184e-04 nat, max |logit difference| = 0.0664.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 14924 | 5.8164 | 5.7500 | 5.8125 | 5.6992 |
| 16309 | 5.1172 | 5.0352 | 5.0664 | 4.9883 |
| 2 | 4.6875 | 4.6172 | 4.6484 | 4.5859 |

## 3b-4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +1.9766; the same difference in candidate-default: +1.9766 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 16.5781 | 16.5781 | 16.5781 | 16.5781 |
| 45647 | 14.6016 | 14.6016 | 14.6016 | 14.6016 |
| 82895 | 14.4375 | 14.4375 | 14.4375 | 14.4375 |

## 3b-4w w1536-gpl-0: parent top-1 11938, candidate top-1 4694 (DIFFER)

Parent-default top-2 margin (logit of id 11938 minus id 4694): +0.0625; the same difference in candidate-default: -0.1875 (moved by -0.2500). KL(P || C) = 8.584e-03 nat, max |logit difference| = 0.5078.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 11938 | 18.3438 | 18.3281 | not run | 18.2656 |
| 4694 | 18.2969 | 18.2656 | not run | 18.4531 |
| 311 | 15.1094 | 15.0469 | not run | 14.9531 |

## 3b-4w w1024-gpl-512: parent top-1 4694, candidate top-1 11938 (DIFFER)

Parent-default top-2 margin (logit of id 4694 minus id 11938): +0.1562; the same difference in candidate-default: -0.0938 (moved by -0.2500). KL(P || C) = 8.050e-03 nat, max |logit difference| = 0.3164.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 4694 | 18.0625 | 18.0469 | not run | 18.0156 |
| 11938 | 17.8438 | 17.8906 | not run | 18.1094 |
| 311 | 14.5859 | 14.5703 | not run | 14.6953 |

## 3b-8da4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 2): +0.3359; the same difference in candidate-default: +0.3477 (moved by +0.0117). KL(P || C) = 5.205e-04 nat, max |logit difference| = 0.2568.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 5.6133 | 5.6250 | 5.6641 | 5.6680 |
| 2 | 5.2422 | 5.2891 | 5.3359 | 5.3203 |
| 14924 | 5.0352 | 5.1562 | 5.2891 | 5.3047 |

## 3b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +1.9922; the same difference in candidate-default: +1.9922 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 16.5625 | 16.5625 | 16.5625 | 16.5625 |
| 45647 | 14.5703 | 14.5703 | 14.5703 | 14.5703 |
| 64130 | 14.4531 | 14.4531 | 14.4531 | 14.4531 |

## 3b-8da4w w1024-gpl-512: parent top-1 4694, candidate top-1 11938 (DIFFER)

Parent-default top-2 margin (logit of id 4694 minus id 11938): +0.3125; the same difference in candidate-default: -1.0000 (moved by -1.3125). KL(P || C) = 2.306e-01 nat, max |logit difference| = 2.7705.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 4694 | 18.1250 | 17.8125 | not run | 18.0625 |
| 11938 | 17.5000 | 17.5000 | not run | 19.0625 |
| 27528 | 14.9375 | 14.3984 | not run | 13.0312 |
| 1234 | 13.1484 | 13.1719 | not run | 14.4062 |

## 8b-4w gate-the2048: parent top-1 3488, candidate top-1 3488 (SAME)

Parent-default top-2 margin (logit of id 3488 minus id 755): +0.1992; the same difference in candidate-default: +0.2305 (moved by +0.0312). KL(P || C) = 4.468e-05 nat, max |logit difference| = 0.0645.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 3488 | 4.8555 | 4.8086 | 4.8633 | 4.8203 |
| 755 | 4.6836 | 4.6094 | 4.6602 | 4.5898 |
| 1839 | 4.4062 | 4.3867 | 4.4062 | 4.3945 |
| 17297 | 4.4727 | 4.3750 | 4.4531 | 4.3477 |

## 8b-4w gate-check1972: parent top-1 6062, candidate top-1 6062 (SAME)

Parent-default top-2 margin (logit of id 6062 minus id 45647): +0.1719; the same difference in candidate-default: +0.1719 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 6062 | 15.2969 | 15.2969 | 15.2969 | 15.2969 |
| 45647 | 15.1250 | 15.1250 | 15.1250 | 15.1250 |
| 70159 | 14.6797 | 14.6797 | 14.6797 | 14.6797 |

## 8b-8da4w gate-the2048: parent top-1 53, candidate top-1 53 (SAME)

Parent-default top-2 margin (logit of id 53 minus id 118): +0.1484; the same difference in candidate-default: +0.3047 (moved by +0.1562). KL(P || C) = 3.211e-03 nat, max |logit difference| = 0.6953.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 53 | 11.3438 | 11.3281 | 11.3828 | 11.4531 |
| 118 | 11.2656 | 11.1797 | 11.1484 | 11.1484 |
| 5531 | 11.0312 | 11.0312 | 11.0859 | 11.1562 |

## 8b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +0.0703; the same difference in candidate-default: +0.0703 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 15.2422 | 15.2422 | 15.2422 | 15.2422 |
| 45647 | 15.1719 | 15.1719 | 15.1719 | 15.1719 |
| 6062 | 15.1719 | 15.1719 | 15.1719 | 15.1719 |

