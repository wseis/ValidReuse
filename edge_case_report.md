# Edge-Case Method Comparison Report

This report summarizes the synthetic edge-case study comparing:

- `Bootstrap`
- `Bayesian approximation`
- `Bayesian (PyMC)`

Each scenario contains 5 synthetic datasets of sample size 12. Reported method statistics are scenario-level means across those 5 replicates.

## Scenario Guide

- `balanced_baseline`: Moderately spread positive vals_zu with ordinary count-like vals_ab and no special pathologies.
- `many_zero_ab`: vals_ab contains many zeros, stressing zero inflation and denominator floor effects after the +1 adjustment.
- `wide_spread_zu`: vals_zu has very large multiplicative spread, creating heavy skew and occasional extreme inflow values.
- `high_outlier_ab`: vals_ab is mostly ordinary counts but includes one very large outlier, testing robustness to extreme output observations.
- `low_signal`: vals_zu is lower and tighter while vals_ab is relatively larger, producing weaker reduction and less separation.
- `bimodal_ab`: vals_ab mixes a near-zero cluster with a much higher cluster, creating a two-mode denominator distribution.

## Method Statistics

| Scenario | Method | Mean L_alpha | Mean Median | Mean Mean | Mean SD |
| --- | --- | --- | --- | --- | --- |
| balanced_baseline | Bootstrap | 6.497 | 6.606 | 6.605 | 0.071 |
| balanced_baseline | Bayesian approximation | 6.425 | 6.584 | 6.578 | 0.087 |
| balanced_baseline | Bayesian (PyMC) | 6.385 | 6.559 | 6.551 | 0.096 |
| many_zero_ab | Bootstrap | 6.892 | 7.117 | 7.106 | 0.133 |
| many_zero_ab | Bayesian approximation | 6.734 | 7.096 | 7.067 | 0.180 |
| many_zero_ab | Bayesian (PyMC) | 6.815 | 7.073 | 7.059 | 0.144 |
| wide_spread_zu | Bootstrap | 6.040 | 6.247 | 6.261 | 0.166 |
| wide_spread_zu | Bayesian approximation | 5.969 | 6.275 | 6.264 | 0.169 |
| wide_spread_zu | Bayesian (PyMC) | 5.763 | 6.192 | 6.165 | 0.223 |
| high_outlier_ab | Bootstrap | 5.714 | 6.487 | 6.356 | 0.348 |
| high_outlier_ab | Bayesian approximation | 5.846 | 6.165 | 6.151 | 0.165 |
| high_outlier_ab | Bayesian (PyMC) | 5.875 | 6.159 | 6.149 | 0.161 |
| low_signal | Bootstrap | 5.081 | 5.156 | 5.156 | 0.052 |
| low_signal | Bayesian approximation | 5.035 | 5.144 | 5.138 | 0.059 |
| low_signal | Bayesian (PyMC) | 4.990 | 5.118 | 5.112 | 0.069 |
| bimodal_ab | Bootstrap | 6.223 | 6.348 | 6.371 | 0.114 |
| bimodal_ab | Bayesian approximation | 6.042 | 6.351 | 6.334 | 0.162 |
| bimodal_ab | Bayesian (PyMC) | 6.031 | 6.337 | 6.322 | 0.162 |

## Lower-Bound Gap Table

| Scenario | Approx - PyMC | Bootstrap - PyMC | Bootstrap - Approx |
| --- | --- | --- | --- |
| balanced_baseline | 0.040 | 0.112 | 0.072 |
| many_zero_ab | -0.080 | 0.077 | 0.158 |
| wide_spread_zu | 0.207 | 0.277 | 0.071 |
| high_outlier_ab | -0.028 | -0.161 | -0.132 |
| low_signal | 0.045 | 0.091 | 0.046 |
| bimodal_ab | 0.010 | 0.192 | 0.182 |

## Median Gap Table

| Scenario | Approx - PyMC | Bootstrap - PyMC | Bootstrap - Approx |
| --- | --- | --- | --- |
| balanced_baseline | 0.025 | 0.047 | 0.022 |
| many_zero_ab | 0.022 | 0.044 | 0.021 |
| wide_spread_zu | 0.083 | 0.055 | -0.029 |
| high_outlier_ab | 0.006 | 0.327 | 0.321 |
| low_signal | 0.026 | 0.038 | 0.012 |
| bimodal_ab | 0.014 | 0.011 | -0.003 |

## Visualizations

### Mean Lower Bound by Scenario

![Mean lower bound comparison](/Users/wseis/Projects/ValidReuse/report_figures/lower_bound_means.png)

### Mean Median by Scenario

![Mean median comparison](/Users/wseis/Projects/ValidReuse/report_figures/median_means.png)

### Lower-Bound Gap Heatmap

![Lower-bound gap heatmap](/Users/wseis/Projects/ValidReuse/report_figures/lower_bound_gap_heatmap.png)

### Median Gap Heatmap

![Median gap heatmap](/Users/wseis/Projects/ValidReuse/report_figures/median_gap_heatmap.png)
