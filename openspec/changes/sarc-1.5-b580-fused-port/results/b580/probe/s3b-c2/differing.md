# Logits at the gate prompts and at every window where the candidate's top-1 differs from the parent's

## 1b-4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 1757): +0.6562; the same difference in candidate-default: +0.6562 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 6.1133 | 6.1172 | 6.1133 | 6.1172 |
| 1757 | 5.4570 | 5.4609 | 5.4570 | 5.4609 |
| 1527 | 5.1562 | 5.1602 | 5.1562 | 5.1602 |

## 1b-4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 11737): +0.5938; the same difference in candidate-default: +0.6406 (moved by +0.0469). KL(P || C) = 3.202e-04 nat, max |logit difference| = 0.1660.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 18.1406 | 18.1406 | 18.1094 | 18.1094 |
| 11737 | 17.5469 | 17.5469 | 17.4688 | 17.4688 |
| 61449 | 17.4219 | 17.4219 | 17.3906 | 17.3906 |

## 1b-8da4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 1757): +0.2578; the same difference in candidate-default: +0.2578 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 5.9375 | 5.8984 | 5.9375 | 5.8984 |
| 1757 | 5.6641 | 5.6406 | 5.6641 | 5.6406 |
| 1527 | 5.5469 | 5.4609 | 5.5469 | 5.4609 |

## 1b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 11737): +0.7812; the same difference in candidate-default: +0.3594 (moved by -0.4219). KL(P || C) = 3.598e-02 nat, max |logit difference| = 1.1250.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 18.1562 | 18.1562 | 18.1406 | 18.1406 |
| 11737 | 17.3750 | 17.3750 | 17.7812 | 17.7812 |
| 40761 | 17.2500 | 17.2500 | 17.5312 | 17.5312 |

## 3b-4w gate-the2048: parent top-1 14924, candidate top-1 14924 (SAME)

Parent-default top-2 margin (logit of id 14924 minus id 16309): +0.7109; the same difference in candidate-default: +0.7109 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 14924 | 5.8125 | 5.6992 | 5.8125 | 5.6992 |
| 16309 | 5.0664 | 4.9883 | 5.0664 | 4.9883 |
| 2 | 4.6484 | 4.5859 | 4.6484 | 4.5859 |

## 3b-4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +1.9766; the same difference in candidate-default: +1.9844 (moved by +0.0078). KL(P || C) = 3.669e-04 nat, max |logit difference| = 0.1791.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 16.5781 | 16.5781 | 16.5781 | 16.5781 |
| 45647 | 14.6016 | 14.6016 | 14.5938 | 14.5938 |
| 82895 | 14.4375 | 14.4375 | 14.3672 | 14.3672 |

## 3b-8da4w gate-the2048: parent top-1 16309, candidate top-1 16309 (SAME)

Parent-default top-2 margin (logit of id 16309 minus id 2): +0.3477; the same difference in candidate-default: +0.3477 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 16309 | 5.6641 | 5.6680 | 5.6641 | 5.6680 |
| 2 | 5.3359 | 5.3203 | 5.3359 | 5.3203 |
| 14924 | 5.2891 | 5.3047 | 5.2891 | 5.3047 |

## 3b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +1.9922; the same difference in candidate-default: +2.3359 (moved by +0.3438). KL(P || C) = 2.217e-02 nat, max |logit difference| = 0.9258.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 16.5625 | 16.5625 | 16.8750 | 16.8750 |
| 45647 | 14.5703 | 14.5703 | 14.5391 | 14.5391 |
| 64130 | 14.4531 | 14.4531 | 14.1094 | 14.1094 |
| 82895 | 14.3672 | 14.3672 | 14.4844 | 14.4844 |

## 8b-4w gate-the2048: parent top-1 3488, candidate top-1 3488 (SAME)

Parent-default top-2 margin (logit of id 3488 minus id 755): +0.2305; the same difference in candidate-default: +0.2305 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 3488 | 4.8633 | 4.8203 | 4.8633 | 4.8203 |
| 755 | 4.6602 | 4.5898 | 4.6602 | 4.5898 |
| 1839 | 4.4062 | 4.3945 | 4.4062 | 4.3945 |
| 17297 | 4.4531 | 4.3477 | 4.4531 | 4.3477 |

## 8b-4w gate-check1972: parent top-1 6062, candidate top-1 6062 (SAME)

Parent-default top-2 margin (logit of id 6062 minus id 45647): +0.1719; the same difference in candidate-default: +0.2188 (moved by +0.0469). KL(P || C) = 3.571e-04 nat, max |logit difference| = 0.1367.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 6062 | 15.2969 | 15.2969 | 15.2812 | 15.2812 |
| 45647 | 15.1250 | 15.1250 | 15.0625 | 15.0625 |
| 70159 | 14.6797 | 14.6797 | 14.6250 | 14.6250 |

## 8b-8da4w gate-the2048: parent top-1 53, candidate top-1 53 (SAME)

Parent-default top-2 margin (logit of id 53 minus id 5531): +0.2969; the same difference in candidate-default: +0.2969 (moved by +0.0000). KL(P || C) = 0.000e+00 nat, max |logit difference| = 0.0000.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 53 | 11.3828 | 11.4531 | 11.3828 | 11.4531 |
| 5531 | 11.0859 | 11.1562 | 11.0859 | 11.1562 |
| 118 | 11.1484 | 11.1484 | 11.1484 | 11.1484 |

## 8b-8da4w gate-check1972: parent top-1 70159, candidate top-1 70159 (SAME)

Parent-default top-2 margin (logit of id 70159 minus id 45647): +0.0703; the same difference in candidate-default: +0.1797 (moved by +0.1094). KL(P || C) = 4.392e-02 nat, max |logit difference| = 1.3711.

| token id | PT (parent tiled) | P (parent default) | CT (candidate tiled) | C (candidate default) |
|---|---:|---:|---:|---:|
| 70159 | 15.2422 | 15.2422 | 15.7734 | 15.7734 |
| 45647 | 15.1719 | 15.1719 | 15.5938 | 15.5938 |
| 6062 | 15.1719 | 15.1719 | 15.0781 | 15.0781 |

