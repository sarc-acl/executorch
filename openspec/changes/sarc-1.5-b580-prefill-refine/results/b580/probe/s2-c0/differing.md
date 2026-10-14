# Logits at the gate prompts and at every window where the candidate's top-1 differs from the parent's

## 1b-4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 1757): +0.6211; the same difference in candidate-default: +0.6328 (moved by +0.0117). KL(P || C) = 1.817e-04 nat, max |logit difference| = 0.0922.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 6.0586 | 6.0547 | 6.0547 | 6.0586 |
| 1757 | 5.4375 | 5.4336 | 5.4258 | 5.4258 |
| 1527 | 5.1094 | 5.1055 | 5.0977 | 5.1016 |

## 1b-4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 11737): +0.5938; the same difference in candidate-default: +0.5938 (moved by +0.0000). KL(P || C) = 1.183e-04 nat, max |logit difference| = 0.0781.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 18.1406 | 18.1406 | 18.1406 | 18.1406 |
| 11737 | 17.5469 | 17.5469 | 17.5469 | 17.5469 |
| 61449 | 17.4219 | 17.4219 | 17.4219 | 17.4219 |

## 1b-4w w1280-gpl-384: parent top-1 304, candidate top-1 311 (DIFFER)

Parent-default top-2 margin (logit of id 304 minus id 311): +0.4062; the same difference in candidate-default: -0.1562 (moved by -0.5625). KL(P || C) = 1.824e-02 nat, max |logit difference| = 0.4688.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 304 | 20.2500 | 20.2812 | not run | 20.0000 |
| 311 | 19.9531 | 19.8750 | not run | 20.1562 |
| 555 | 19.7188 | 19.6562 | not run | 19.5938 |

## 1b-8da4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 1757): +0.2227; the same difference in candidate-default: +0.1953 (moved by -0.0273). KL(P || C) = 4.117e-03 nat, max |logit difference| = 0.3594.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 6.1250 | 5.9023 | 5.7500 | 5.8359 |
| 1757 | 5.8594 | 5.6797 | 5.5312 | 5.6406 |
| 1527 | 5.7539 | 5.5273 | 5.2969 | 5.4609 |

## 1b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 40761): +0.1875; the same difference in candidate-default: +0.9062 (moved by +0.7188). KL(P || C) = 2.826e-02 nat, max |logit difference| = 1.0078.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 17.8125 | 17.8125 | 18.1562 | 18.1562 |
| 40761 | 17.6250 | 17.6250 | 17.2500 | 17.2500 |
| 11737 | 17.4375 | 17.4375 | 17.3750 | 17.3750 |

## 1b-8da4w w1536-gpl-0: parent top-1 539, candidate top-1 11 (DIFFER)

Parent-default top-2 margin (logit of id 539 minus id 11): +0.3906; the same difference in candidate-default: -0.9844 (moved by -1.3750). KL(P || C) = 2.194e-01 nat, max |logit difference| = 4.9941.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 539 | 19.1875 | 19.5469 | not run | 18.6094 |
| 11 | 18.8750 | 19.1562 | not run | 19.5938 |
| 320 | 15.8125 | 17.1250 | not run | 16.8750 |
| 4694 | 17.4531 | 15.8125 | not run | 16.4844 |

## 3b-4w gate-the2048: parent top-1 14924, candidate top-1 14924 (SAME)

Parent-default top-2 margin (logit of id 14924 minus id 16309): +0.3945; the same difference in candidate-default: +0.7148 (moved by +0.3203). KL(P || C) = 1.522e-01 nat, max |logit difference| = 2.1133.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 14924 | 7.3906 | 7.3047 | 5.8164 | 5.7500 |
| 16309 | 6.9766 | 6.9102 | 5.1172 | 5.0352 |
| 3936 | 6.3164 | 6.2500 | 4.2266 | 4.1367 |
| 2 | 6.1992 | 6.1523 | 4.6875 | 4.6172 |

## 3b-4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +1.9688; the same difference in candidate-default: +1.9766 (moved by +0.0078). KL(P || C) = 7.270e-05 nat, max |logit difference| = 0.0752.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 16.5781 | 16.5781 | 16.5781 | 16.5781 |
| 45647 | 14.6094 | 14.6094 | 14.6016 | 14.6016 |
| 82895 | 14.4531 | 14.4531 | 14.4375 | 14.4375 |

## 3b-4w w1536-gpl-0: parent top-1 4694, candidate top-1 11938 (DIFFER)

Parent-default top-2 margin (logit of id 4694 minus id 11938): +0.1250; the same difference in candidate-default: -0.0625 (moved by -0.1875). KL(P || C) = 6.106e-03 nat, max |logit difference| = 0.5703.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 4694 | 18.4062 | 18.3438 | not run | 18.2656 |
| 11938 | 18.1562 | 18.2188 | not run | 18.3281 |
| 311 | 15.1484 | 15.3438 | not run | 15.0469 |

## 3b-8da4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 3936): +0.4727; the same difference in candidate-default: +0.5781 (moved by +0.1055). KL(P || C) = 1.825e-01 nat, max |logit difference| = 2.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 7.5938 | 7.5195 | 5.6133 | 5.6250 |
| 3936 | 7.1172 | 7.0469 | 5.0117 | 5.0469 |
| 2 | 7.0547 | 7.0078 | 5.2422 | 5.2891 |
| 14924 | 4.7852 | 4.7734 | 5.0352 | 5.1562 |

## 3b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +1.9375; the same difference in candidate-default: +1.9922 (moved by +0.0547). KL(P || C) = 1.220e-02 nat, max |logit difference| = 0.7969.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 16.4375 | 16.4375 | 16.5625 | 16.5625 |
| 45647 | 14.5000 | 14.5000 | 14.5703 | 14.5703 |
| 82895 | 14.3750 | 14.3750 | 14.3672 | 14.3672 |
| 64130 | 13.9922 | 13.9922 | 14.4531 | 14.4531 |

## 3b-8da4w w1536-gpl-0: parent top-1 4694, candidate top-1 11938 (DIFFER)

Parent-default top-2 margin (logit of id 4694 minus id 11938): +0.3438; the same difference in candidate-default: -0.5938 (moved by -0.9375). KL(P || C) = 1.569e-01 nat, max |logit difference| = 2.7188.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 4694 | 19.0312 | 18.0938 | not run | 18.9688 |
| 11938 | 16.9375 | 17.7500 | not run | 19.5625 |
| 311 | 14.9453 | 15.5625 | not run | 15.0469 |

## 8b-4w gate-the2048: parent top-1 3488, candidate top-1 3488 (SAME)

Parent-default top-2 margin (logit of id 3488 minus id 17297): +0.0508; the same difference in candidate-default: +0.4336 (moved by +0.3828). KL(P || C) = 4.170e-02 nat, max |logit difference| = 1.7617.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 3488 | 5.8750 | 5.7148 | 4.8555 | 4.8086 |
| 17297 | 5.8047 | 5.6641 | 4.4727 | 4.3750 |
| 755 | 5.5312 | 5.3828 | 4.6836 | 4.6094 |
| 1839 | 4.3750 | 4.3828 | 4.4062 | 4.3867 |

## 8b-4w gate-check1972: parent top-1 6062, candidate top-1 6062 (SAME)

Parent-default top-2 margin (logit of id 6062 minus id 45647): +0.2031; the same difference in candidate-default: +0.1719 (moved by -0.0312). KL(P || C) = 1.243e-04 nat, max |logit difference| = 0.0859.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 6062 | 15.3203 | 15.3203 | 15.2969 | 15.2969 |
| 45647 | 15.1172 | 15.1172 | 15.1250 | 15.1250 |
| 70159 | 14.6953 | 14.6953 | 14.6797 | 14.6797 |

## 8b-8da4w gate-the2048: parent top-1 247, candidate top-1 53 (DIFFER)

Parent-default top-2 margin (logit of id 247 minus id 118): +0.3359; the same difference in candidate-default: -2.2500 (moved by -2.5859). KL(P || C) = 9.751e-01 nat, max |logit difference| = 6.0640.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 247 | 11.5156 | 11.6641 | 9.0625 | 8.9297 |
| 118 | 11.1094 | 11.3281 | 11.2656 | 11.1797 |
| 246 | 10.7500 | 11.0000 | 10.0625 | 10.2500 |
| 53 | 8.4375 | 8.1250 | 11.3438 | 11.3281 |
| 5531 | 10.3281 | 10.2500 | 11.0312 | 11.0312 |

## 8b-8da4w gate-check1972: parent top-1 45647, candidate top-1 70159 (DIFFER)

Parent-default top-2 margin (logit of id 45647 minus id 6062): +0.1484; the same difference in candidate-default: +0.0000 (moved by -0.1484). KL(P || C) = 4.212e-02 nat, max |logit difference| = 1.4473.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 45647 | 15.4688 | 15.4688 | 15.1719 | 15.1719 |
| 6062 | 15.3203 | 15.3203 | 15.1719 | 15.1719 |
| 70159 | 15.0000 | 15.0000 | 15.2422 | 15.2422 |

