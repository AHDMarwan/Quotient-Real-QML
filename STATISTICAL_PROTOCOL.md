# Statistical study protocol

This study tests whether quotient-aware removal of hidden gauge directions improves finite-shot estimation and optimization under equal total shot budgets.

## Gradient-noise study

- 500 paired random seeds per budget.
- 100 Monte-Carlo gradient estimates per physical parameter and seed.
- Total parameter-shift shot budgets: 6,000; 12,000; 24,000; 48,000.
- Primary outcome: physical-gradient mean-squared error.

## Noisy-convergence study

- 200 paired random seeds per budget.
- 30 optimization epochs per seed.
- Total shot budgets per epoch: 750; 1,500; 3,000; 6,000.
- Same teacher, data, initialization, optimizer, and physical circuit within each paired seed.
- Primary outcome: mean exact MSE over the final optimization tail.

## Inference

For each budget the analysis reports:

- paired bootstrap 95% confidence intervals from 4,000 resamples;
- paired two-sided t-test;
- Wilcoxon signed-rank test;
- Holm multiplicity correction across the four budgets within each study;
- paired Cohen dz effect size;
- Wilson 95% interval for the fraction of paired trials where the augmented parameterization is worse.

A fixed-model advantage is flagged only when the paired mean-difference bootstrap interval is strictly above zero and at least one Holm-adjusted paired test is below 0.05. Gradient-noise and convergence conclusions are reported separately.

All random-number streams are deterministically seeded, and GitHub Actions records the exact code revision and uploads both summary and seed-level trial tables for reproducibility.
