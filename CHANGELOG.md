# Changelog

## [v4.0.0](https://github.com/Lsh0x/rs-stats/tree/v4.0.0)

The "trust the tails" release: every numeric path is cross-validated
against scipy/numpy, including extreme-tail regimes, and the whole
library gets sampling, survival functions, non-parametric tests,
resampling, exact power analysis and 5 new distributions.

**Breaking changes:**

- `chi_square_goodness_of_fit` / `chi_square_independence` return a named
  `ChiSquareResult { statistic, degrees_of_freedom, p_value }` instead of
  a positional `(f64, usize, f64)` tuple.
- `std_err` now uses the **sample** convention (ddof = 1, matching
  `scipy.stats.sem`); the previous population behaviour is available as
  `std_err_population`.
- Degenerate inputs that silently returned `NaN`/`±∞` are now typed
  errors: zero-variance t-tests and ANOVA, `z_score` with σ ≤ 0, all-zero
  χ² tables, `p = 1.0` quantiles of unbounded discrete distributions,
  non-finite distribution parameters in every `new()`, NaN data in
  `describe`/`quantile`, correlation and rank tests.
- `Exponential` out-of-support behaviour aligned with the other continuous
  distributions: `pdf(x<0) = 0`, `cdf(x<0) = 0`, `logpdf(x<0) = −∞`
  (previously the lone `Err`).
- Public structs gained fields (`LinearRegression::{x_mean, sum_xx}`,
  `MultipleLinearRegression` per-coefficient inference,
  `DecisionTree` internals); struct literals from older versions no longer
  compile. Saved serde models from v3 still load (`serde(default)`).
- `LinearRegression::confidence_interval` accepts any level in (0, 1) and
  now returns a statistically correct Student-t interval of the mean
  response (hardcoded normal z-scores are gone).
- `Welford::population_variance` is deprecated in favour of
  `variance_population`; `variance_sample` added.
- `DecisionTree` no longer requires `Eq + Hash` on targets
  (`DecisionTree<f64, f64>` now compiles); the MAE criterion evaluates
  and predicts medians (its actual minimiser) instead of means.
- Cargo features introduced: `parallel` (rayon) and `serde`
  (serde/serde_json/bincode), both enabled by default;
  `--no-default-features` builds with only `num-traits` + `rand`.
  Unused `num` dependency removed; `rand_chacha` moved to
  dev-dependencies; `thiserror` upgraded to 2.

**Added:**

- `sample()` / `sample_n()` on all distributions: ziggurat Normal /
  LogNormal, Marsaglia-Tsang Gamma (shared by Beta, χ², Student-t, F),
  Knuth/PTRS Poisson, BINV Binomial, Gamma-Poisson NegativeBinomial.
- Exact survival functions `sf(x)` on all distributions (full relative
  precision where `1 − cdf` collapses to 0).
- New distributions: Cauchy, Laplace, Pareto, Logistic (14 auto-fit
  candidates) and `MultivariateNormal` (Cholesky-validated, scipy-exact
  logpdf, Mahalanobis score, `μ + L·z` sampling).
- Hypothesis tests: Mann-Whitney U, Wilcoxon signed-rank, two-sample
  Kolmogorov-Smirnov, Fisher exact, D'Agostino-Pearson K² normality test,
  one-sided `Alternative`s on t-tests, `cohens_d`, `AnovaResult::eta_squared`,
  `p_adjust` (Bonferroni / Holm / Benjamini-Hochberg).
- `resampling`: `bootstrap_ci` (percentile) and `permutation_test` for any
  statistic.
- Exact power analysis: `power_t_test` / `sample_size_t_test` via a new
  noncentral-t CDF (Lenth AS 243).
- Correlation: `pearson` / `spearman` (tie-aware) with significance tests.
- Descriptive: `describe()` and numpy-compatible `quantile()`.
- Regression: `prediction_interval`, per-coefficient SE/t/p and global
  F-test in multiple regression.
- `utils::linalg::cholesky`; `ln_permutation` / `ln_combination`.

**Fixed:**

- Deep-tail correctness: dynamic quantile brackets (StudentT ν < 2,
  F(1,1)), `erfc` exact in the far tail, Binomial pmf/cdf at large n,
  Poisson `ln(k!)` beyond the table, u64-edge combinatorics,
  `feature_importances` panic, sub-1e-12 quantiles of tiny-shape
  Gamma/Beta (log-scale root bracketing).
- Performance: Cody rational erf/erfc, O(1) discrete CDFs, single-pass
  fit_all diagnostics, Newton quantiles, incremental decision-tree splits
  (4–25×), multi-accumulator reductions, rayon size thresholds.

## [v3.0.0](https://github.com/Lsh0x/rs-stats/tree/v3.0.0)

**Breaking changes** — slimmed-down public surface and rayon-default
parallelism. Internal math is unchanged; users on the trait API
(`Normal::new(…).pdf(…)`, `auto_fit`, `fit_all`, etc.) need only update
the version. Users who imported the legacy free functions or `*Config<T>`
types will need to migrate as described below.

**Removed (breaking):**

- `*Config<T>` types: `NormalConfig`, `PoissonConfig`, `BinomialConfig`,
  `ExponentialConfig`, `UniformConfig`. Use `Normal::new(μ, σ)` etc. directly.
- Public free functions: `normal_pdf<T>`, `normal_cdf<T>`,
  `normal_inverse_cdf<T>`, `poisson::pmf<T>`, `poisson::cdf<T>`,
  `binom::pmf<T>`, `binom::cdf<T>`, `uniform_{pdf,cdf,inverse_cdf,mean}<T>`,
  `exponential_{pdf,cdf,inverse_cdf,mean,variance}<T>`. The same maths
  is reachable via the `Distribution` / `DiscreteDistribution` trait
  impls on the typed structs (`Normal::new(μ, σ).pdf(x)`).
- `prob::probability_density`, `prob::cumulative_distrib`,
  `prob::normal_cumulative_distrib`, `prob::normal_probability_density`
  (and the corresponding `prob_density` / `cumulative_distrib` /
  `normal_cumulative_distrib` modules). All redundant with `Normal::pdf` /
  `Normal::cdf` on the typed struct.

**Migration cheatsheet:**

```text
v2.x                                              v3.0
────────────────────────────────────────────────────────────────────────
normal_pdf(x, μ, σ)                              Normal::new(μ, σ)?.pdf(x)
normal_cdf(x, μ, σ)                              Normal::new(μ, σ)?.cdf(x)
normal_inverse_cdf(p, μ, σ)                      Normal::new(μ, σ)?.inverse_cdf(p)
prob::cumulative_distrib(x, μ, σ)                Normal::new(μ, σ)?.cdf(x)
prob::probability_density(x, μ, σ)               Normal::new(μ, σ)?.pdf(x)
poisson::pmf(k, λ)                               Poisson::new(λ)?.pmf(k)
poisson::cdf(k, λ)                               Poisson::new(λ)?.cdf(k)
binom::pmf(k, n, p)                              Binomial::new(n, p)?.pmf(k)
binom::cdf(k, n, p)                              Binomial::new(n, p)?.cdf(k)
NormalConfig { mean, std_dev }                   Normal { mean, std_dev }
NormalConfig::new(μ, σ)                          Normal::new(μ, σ)
PoissonConfig { lambda }                         Poisson { lambda }
…                                                (same pattern for the
                                                  4 other Configs)
```

**Performance:**

- `rayon` is now a default dependency (no longer opt-in via the
  `parallel` feature). Callers who need single-threaded execution can
  configure rayon's global thread pool with
  `rayon::ThreadPoolBuilder::new().num_threads(1).build_global()`.
- `fit_all`, `fit_all_verbose`, `fit_all_discrete`, `fit_all_discrete_verbose`
  now run their candidate fits in parallel (10-way / 4-way). Each
  candidate (Normal::fit, LogNormal::fit, Gamma::fit, …) lands on its
  own rayon worker.
- `one_way_anova` walks each group's Welford pass on its own worker.
- `LinearRegression::predict_many` and decision-tree split search were
  already conditionally parallel; the cfg gates are now gone, so they
  always parallelise.

**Trait merge (breaking):**

- `Distribution` and `DiscreteDistribution` are now a single
  [`Distribution`] trait parameterised by an associated support type
  `X` (`f64` for continuous, `u64` for discrete). `Box<dyn Distribution>`
  becomes `Box<dyn Distribution<X = f64>>` (or `… X = u64 …`).
- `DiscreteDistribution` is kept as a back-compat blanket alias
  (`pub trait DiscreteDistribution: Distribution<X = u64>`) so existing
  imports keep working — `pmf` / `logpmf` likewise survive as default
  trait methods that delegate to `pdf` / `logpdf`.
- `fit_all*` and `fit_all_discrete*` use one shared registration
  pattern internally. Adding a new distribution = one line in the
  fitter array.
- `Distribution::log_likelihood_fast(&self, &[X]) -> f64` — new
  infallible bulk log-likelihood. Out-of-support points contribute
  `f64::NEG_INFINITY` (scipy convention). Default impl walks `logpdf`
  in a loop; `Normal` overrides with a closed-form
  `−½ · Σ ((xᵢ−μ)/σ)²` reduction that LLVM can autovectorise.

**Variance / std-dev (breaking name additions):**

- New: `prob::variance_population` / `prob::variance_sample` and
  `prob::std_dev_population` / `prob::std_dev_sample`. Match the numpy
  / scipy naming exactly:
  - `*_population` ↔ `numpy.var(data, ddof=0)`
  - `*_sample` ↔ `numpy.var(data, ddof=1)` / `pandas.Series.var()`
- `prob::variance` and `prob::std_dev` keep their v2.x behaviour
  (population) and are now documented aliases.

**Performance:**

- `MultipleLinearRegression` — internal storage is now flat row-major
  `Vec<T>` instead of `Vec<Vec<T>>`. The augmented design matrix is
  built directly from the input rows, no intermediate `Vec<Vec<T>>`
  copy. New flat-friendly helpers `matrix_multiply_transpose_flat` /
  `vector_multiply_transpose_flat` / `solve_linear_system_flat`.
  Reduces n+1 heap allocations to one and keeps inner loops
  contiguous for autovec.
- `DecisionTree::find_best_split` rewritten to use a single
  sort-then-prefix-slice strategy: per (node, feature) we now sort
  indices once and walk thresholds with prefix / suffix views of the
  sorted vector. The previous code rebuilt `left_indices` /
  `right_indices` `Vec`s for every threshold — O(features × thresholds)
  fresh allocations per node. v3.0 materialises the index vectors only
  for the **best** split, dropping inner-loop allocations to zero.

**Internals (non-breaking, but visible to readers):**

- The previous macro-based `try_fit!` registration in `fitting.rs` is
  replaced by typed `fn` arrays (`CONTINUOUS_FITTERS`,
  `CONTINUOUS_VERBOSE_FITTERS`, `DISCRETE_FITTERS`,
  `DISCRETE_VERBOSE_FITTERS`). Cleaner par-iter dispatch, easier to add
  a new distribution.
- All math helpers in `normal_distribution.rs`, `poisson_distribution.rs`,
  `binomial_distribution.rs`, `uniform_distribution.rs`,
  `exponential_distribution.rs` now take `f64` directly instead of being
  generic over `T: ToPrimitive`. Validation is centralised in `*::new()`.

**Net:** ~1400 LOC of duplicate API surface deleted. 330 unit + 68 doc +
35 validation tests still pass.

[Full Changelog](https://github.com/Lsh0x/rs-stats/compare/v2.1.0...v3.0.0)

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
