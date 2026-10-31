# Benchmark Results

This file records manual benchmark comparisons run during development.

## DIM100 | PLS-12 | N_train=200 | Order=3 | q=0.8 | theta optimize = true

Validation set: `N_val=2000`

| Model | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APC | 0.3182 | 0.0142 | 0.3324 | 0.2379 | 0.0722 | OK |
| APCK | 58.3982 | 0.2174 | 58.6155 | 0.2416 | 0.0509 | OK |
| GEAPCK | 374.8529 | 0.2427 | 375.0957 | 0.3198 | 0.1065 | OK |
| GEAPCK_BLOCK | 106.7725 | 0.4400 | 107.2124 | 0.2371 | 0.0615 | OK |

Notes:
- `GEAPCK_BLOCK` LOO was corrected from an optimistic pseudo-LOO to a backbone-based approximation before this run.
- Prediction path was optimized to avoid repeated derivative-kernel recomputation per block before this run.

## DIM100 | PLS-15 | N_train=200 | Order=3 | q=0.8 | theta optimize = true

Validation set: `N_val=2000`

| Model | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APC | 0.3117 | 0.0151 | 0.3268 | 0.2664 | 0.0708 | OK |
| APCK | 118.7986 | 0.4175 | 119.2161 | 0.2658 | 0.0450 | OK |
| GEAPCK_BLOCK | 314.2135 | 0.8251 | 315.0386 | 0.2733 | 0.0554 | OK |

GEAPCK_BLOCK `num_blocks` sweep under the same setting:

| Blocks | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 470.0855 | 0.9002 | 470.9856 | 0.2733 | 0.0554 | OK |
| 3 | 355.0951 | 0.7189 | 355.8140 | 0.2733 | 0.0554 | OK |
| 5 | 260.5815 | 0.7373 | 261.3188 | 0.2733 | 0.0554 | OK |

Notes:
- These rounded values look identical in `RelMSE/LOO`, but deeper inspection showed the predictions were not exactly identical; the changes were only very small at the aggregate metric level.

## DIM150 | PLS-18 | N_train=120 | Order=3 | q=0.8 | theta optimize = true

Validation set: `N_val=2000`

| Model | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APC | 0.3158 | 0.0157 | 0.3315 | 0.3006 | 3.371e-04 | OK |
| APCK | 128.6567 | 0.2647 | 128.9214 | 0.2898 | 1.241e-04 | OK |
| GEAPCK | 1035.3880 | 0.2571 | 1035.6451 | 0.6669 | 0.8370 | OK |
| GEAPCK_BLOCK | 155.3663 | 0.4643 | 155.8306 | 0.3000 | 7.041e-04 | OK |

## DIM150 | PLS-20 | N_train=200 | Order=3 | q=0.8 | theta optimize = true

Validation set: `N_val=2000`

| Model | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APC | 0.3247 | 0.0164 | 0.3411 | 0.1721 | 8.739e-03 | OK |
| APCK | 208.6910 | 0.5124 | 209.2035 | 0.1497 | 4.472e-03 | OK |
| GEAPCK_BLOCK | 901.0878 | 1.0392 | 902.1271 | 0.1237 | 7.457e-03 | OK |

Notes:
- This run intentionally excluded `GEAPCK` per request.

## WINGWEIGHT | No Reduction | Order=4 | q=0.9 | N_train=200 | trend_range=[0.2,0.8] | theta optimize = true

Validation set: `N_val=2000`

| Model | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APC | 0.3250 | 0.0162 | 0.3413 | 5.648e-05 | 1.229e-04 | OK |
| APCK | 285.5237 | 0.1592 | 285.6829 | 3.172e-05 | 3.513e-06 | OK |
| GEAPCK | 448.5032 | 0.1943 | 448.6975 | 3.442e-08 | 6.237e-09 | OK |
| GEAPCK_BLOCK(2) | 508.6983 | 0.2962 | 508.9945 | 1.393e-06 | 4.892e-06 | OK |
| GEAPCK_BLOCK(3) | 422.3152 | 0.3070 | 422.6222 | 1.686e-06 | 4.991e-06 | OK |
| GEAPCK_BLOCK(4) | 401.9570 | 0.3323 | 402.2892 | 3.855e-06 | 4.468e-06 | OK |
| GEAPCK_BLOCK(5) | 365.3690 | 0.3691 | 365.7381 | 5.634e-06 | 4.960e-06 | OK |

## DIM150 | PLS-18 | N_train=200 | Order=3 | q=0.8 | trend_range=[0.2,0.6] | theta optimize = true

Validation set: `N_val=2000`

| Model | Train(s) | Pred(s) | Total(s) | RelMSE | LOO | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APC | 0.4041 | 0.0157 | 0.4198 | 0.1295 | 8.552e-03 | OK |
| APCK | 105.1339 | 0.4307 | 105.5646 | 0.1215 | 5.144e-03 | OK |
| GEAPCK | 3218.8800 | 0.4940 | 3219.3740 | 0.2129 | 5.733e-02 | OK |
| GEAPCK_BLOCK(4) | 411.8107 | 0.9499 | 412.7606 | 0.1213 | 7.127e-03 | OK |
