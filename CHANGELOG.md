# Changelog

## [v2.1.0](https://github.com/Lsh0x/rs-stats/tree/v2.1.0)

**License:** the project is now MIT-licensed (was GPL-3.0).

**New:**

- `prob::welford` — numerically stable online estimators: `Welford` (scalar),
  `WelfordVector` (per-axis), `WelfordCovariance` (full covariance matrix
  with zero per-call allocation via persistent scratch buffer).
  `Welford::pop` for subtractive updates, `merge` (Chan 1979) on all three
  for distributed accumulation.
- `utils::linalg` — small dense linear algebra: `invert` /
  `invert_with_ridge` (Gauss-Jordan with partial pivoting, single shared
  augmented-matrix kernel), `mahalanobis_sq` and zero-allocation
  `mahalanobis_sq_into` variant for hot per-row scoring loops.
- `WelfordVector::variance_into` / `std_dev_into` — zero-allocation
  per-axis variants.
- `distributions::fitting::ks_test_with_scratch` /
  `ks_test_discrete_with_scratch` — caller-provided buffer variants of the
  Kolmogorov-Smirnov tests; `fit_all` / `fit_all_verbose` /
  `fit_all_discrete` / `fit_all_discrete_verbose` now reuse a single
  scratch buffer instead of allocating one per candidate.

**Fixed bugs:**

- `t_test::{one_sample, two_sample (Welch), paired}_t_test` returned
  incorrect p-values (often `1.0`) due to a broken in-file
  `incomplete_beta`. Now delegates to the canonical
  `utils::special_functions::regularized_incomplete_beta`.
- `anova::one_way_anova` returned `0.0` p-values for moderate F-statistics
  because of a buggy in-file rational approximation. Now uses the
  canonical incomplete beta.
- `chi_square::{goodness_of_fit, independence}` used the Wilson-Hilferty
  normal approximation (~0.3 % error). Now uses the canonical
  `regularized_incomplete_gamma` for the χ² survival function (matches
  `scipy.stats.chi2.sf` to ~1e-12).
- `normal_inverse_cdf` returned wildly incorrect values in the lower / upper
  tails (p ≤ 0.02425, p ≥ 0.97575). Acklam's rational approximation is now
  applied as published, without the spurious `-t` shift.
- `prob::erf` upgraded to track scipy to ~1e-12 (was ~1.5e-7 max error
  via Abramowitz-Stegun 7.1.26). Internally delegates to
  `regularized_incomplete_gamma(0.5, x²)`.

**Performance:**

- `LogNormal::fit` — single-pass online mean+variance on `ln(x)` instead
  of allocating an intermediate `Vec<f64>`.
- `linalg::invert_with_ridge` — builds the augmented `[A+λI | I]` matrix
  in one pass, removing the prior `matrix.to_vec()` clone (saves one
  `Vec<f64>` of length `dim²` per call).
- `fit_all` and friends — pre-allocated capacity hints + shared KS
  scratch buffer (10× fewer allocations on the continuous fit path,
  4× fewer on the discrete one).

**Validation:**

- All distribution PDF/CDF/PPF, hypothesis test statistics + p-values,
  Welford estimators, special functions and linalg primitives are now
  cross-validated against numpy / scipy / sklearn via a local harness in
  `validation/` (gitignored). 35 invariants tracked; matches scipy to
  1e-9 / 1e-12 on most paths.

[Full Changelog](https://github.com/Lsh0x/rs-stats/compare/v2.0.3...v2.1.0)

## [Unreleased](https://github.com/Lsh0x/rs-stats/tree/HEAD)

[Full Changelog](https://github.com/Lsh0x/rs-stats/compare/3b6164864800773f1e475b62ec24a04d2cc76930...HEAD)

**Fixed bugs:**

- Fix t-test p-value calculation producing values > 1.0. The p-value is now correctly clamped to [0.0, 1.0] range.

**Closed issues:**

- ERF error  [\#3](https://github.com/Lsh0x/rs-stats/issues/3)

**Merged pull requests:**

- feat: add decision tree to regression [\#19](https://github.com/Lsh0x/rs-stats/pull/19) ([Lsh0x](https://github.com/Lsh0x))
- fix: cron for sync template [\#18](https://github.com/Lsh0x/rs-stats/pull/18) ([Lsh0x](https://github.com/Lsh0x))
- feat: create pull request to sync rbase [\#17](https://github.com/Lsh0x/rs-stats/pull/17) ([Lsh0x](https://github.com/Lsh0x))
- feat: update codecove version [\#16](https://github.com/Lsh0x/rs-stats/pull/16) ([Lsh0x](https://github.com/Lsh0x))
- feat: linear regression [\#15](https://github.com/Lsh0x/rs-stats/pull/15) ([Lsh0x](https://github.com/Lsh0x))
- doc: rewrite readme and increase version to 1.0.2 [\#13](https://github.com/Lsh0x/rs-stats/pull/13) ([Lsh0x](https://github.com/Lsh0x))
- doc: rewrite readme [\#12](https://github.com/Lsh0x/rs-stats/pull/12) ([Lsh0x](https://github.com/Lsh0x))
- fix: add cargo lock [\#11](https://github.com/Lsh0x/rs-stats/pull/11) ([Lsh0x](https://github.com/Lsh0x))
- feat: rework tree and add distribution hypothesis and prod dir [\#10](https://github.com/Lsh0x/rs-stats/pull/10) ([Lsh0x](https://github.com/Lsh0x))
- feat: update readme [\#9](https://github.com/Lsh0x/rs-stats/pull/9) ([Lsh0x](https://github.com/Lsh0x))
- fix: export properly lib func [\#8](https://github.com/Lsh0x/rs-stats/pull/8) ([Lsh0x](https://github.com/Lsh0x))
- feat: bump version to 0.2.0 [\#7](https://github.com/Lsh0x/rs-stats/pull/7) ([Lsh0x](https://github.com/Lsh0x))
- rf: split sources into different files [\#6](https://github.com/Lsh0x/rs-stats/pull/6) ([Lsh0x](https://github.com/Lsh0x))
- fix: erf neg infinity and last commit  [\#5](https://github.com/Lsh0x/rs-stats/pull/5) ([Lsh0x](https://github.com/Lsh0x))
- fix: ci doc and coverage [\#4](https://github.com/Lsh0x/rs-stats/pull/4) ([Lsh0x](https://github.com/Lsh0x))
- fix: github action workflow [\#2](https://github.com/Lsh0x/rs-stats/pull/2) ([Lsh0x](https://github.com/Lsh0x))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
