# Review of the updated ValidReuse Robustness Review

Reviewed the 17-page PDF dated 9 September 2026, 10:31, supplied at
`/Users/wseis/Downloads/ValidReuse Robustness Review.pdf`. Its final page identifies
code version `efc335d`; implementation started from `6fe2e1ec`. Numerical tables
in the PDF were not independently reproduced in full.

## Findings incorporated in the count/volume workflow

- Pages 8–10: analytical volume affects what can be measured. Individual effective
  volumes are now mandatory for count analysis; recovery, dilution and analysed
  fraction are explicit. No constant 100 mL volume is silently assumed.
- Pages 7–8 and 13–14: chain diagnostics are essential. The new model branch returns
  unrounded R-hat, tail ESS, divergences and Monte Carlo uncertainty of the lower
  quantile bound. A flagged result is not labelled as meeting the model criterion.
- Pages 15–17: retain Poisson-lognormal and negative-binomial candidates, check
  predictive zero frequency, maximum, q90 and dispersion, and report model spread.
  Both models run against the same exposure-aware count likelihood. The PLN is
  listed first; no automated winner selection is used.
- Pages 10–12: no automatic deletion or outlier downweighting. Technical sample
  identities are preserved and missing entries cannot shift count-volume alignment.
- Page 6: display results to two decimals, accompanied by numerical diagnostics,
  rather than suggesting four-decimal scientific accuracy. Full values remain in
  machine-readable export.

## Qualifications to the document

1. An exact raw count of zero is not a censored positive concentration. It has
   probability exp(-E*C) under the Poisson sampling model. Generic <LOD, <LOQ and
   MPN reports require other observation models; they are not silently treated as
   zero or integer counts.
2. Seed variation at finite bootstrap sizes does not establish failure of
   asymptotic convergence. Discrete support, near-boundary cumulative mass, and the
   nested simulation need separate examination. The bootstrap does resample the
   frequency of an observed extreme value, so the statement that every procedure
   treats the observed 5% bad-day rate as known is too strong. It does not, however,
   explicitly estimate an independently defined operational-regime probability.
3. The old unpaired bootstrap forms ratios from independently sampled marginals.
   Its target is not literally the empirical quantile of 20 observed paired LRVs;
   the “two worst measurements” explanation is an intuition, not an exact account
   of that algorithm. A nonsignificant sample correlation cannot prove independence.
4. The displayed increasing-N table is not monotone (for example bootstrap lower
   bound rises from 6.093 at N=12 to 6.151 at N=16), and the prefixes change the data
   composition. It does not isolate sample-size bias or prove universal optimism.
5. Better PLN fit in one dataset does not establish that it is always more
   conservative or that lognormal influent forces lognormal effluent. The app
   treats both as candidates. The latent-concentration LRV introduced here differs
   from the review's count-predictive, zero-adjusted LRV, so numerical agreement with
   the PDF is not an acceptance criterion for this new workflow.
6. The volume effect applies to effective analysed original water, not merely
   collected/filtered litres. 15 L filtered with the guideline's 25% analysed
   fraction and 50% recovery is 1.875 L effective. General volume-dependent
   measurement envelopes cannot be reduced to one fixed influent-only ceiling.
7. Combining barrier medians and variances requires assumptions about dependence
   and distributions; the review's brief recommendation is not a general rule.

## Deliberate scope limits

The new candidates estimate the population quantile of latent LRV under explicitly
acknowledged independent influent and effluent distributions. They retain pairing
metadata but do not fit correlated paired processes. Recovery and volumes are
fixed inputs; uncertainty propagation, censored/MPN likelihoods, automated
leave-one-sample-out analysis, temporal models and full simulation calibration
remain further work. The existing calculator is preserved as a clearly marked
legacy workflow, rather than feeding unequal-volume counts into its old models.
