# rs-stats

[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/lsh0x/rs-stats/workflows/CI/badge.svg)](https://github.com/lsh0x/rs-stats/actions)
[![Docs](https://docs.rs/rs-stats/badge.svg)](https://docs.rs/rs-stats)
[![Crates.io](https://img.shields.io/crates/v/rs-stats.svg)](https://crates.io/crates/rs-stats)

**Statistics for Rust that you can actually trust in the tails.**

14 parametric distributions with a unified `pdf / cdf / sf / quantile / sample` interface, automatic distribution fitting, parametric *and* non-parametric hypothesis tests, correlation, regression with real confidence intervals — every numeric path cross-validated against scipy/numpy, including the extreme-tail regimes where naive implementations silently return `0.0`.

```toml
[dependencies]
rs-stats = "4"
```

---

## Sixty seconds of rs-stats

**"What distribution is my data, and how extreme is this new observation?"**

```rust
use rs_stats::{auto_fit, Distribution};
use rs_stats::distributions::lognormal::LogNormal;

// Response times (ms) from a production service — right-skewed, as always.
let latencies = vec![
    12.1, 14.8, 15.2, 17.9, 19.3, 22.4, 25.1, 28.7, 33.0, 41.2,
    48.9, 55.3, 71.8, 13.4, 16.0, 18.2, 12.9, 14.1, 96.5, 24.6,
];

// One call: detect the type, fit 10 candidate distributions in parallel,
// rank them by AIC. → LogNormal wins.
let best = auto_fit(&latencies).unwrap();
println!("best fit: {} (AIC = {:.1})", best.name, best.aic);

// Fit it explicitly and ask real questions.
let dist = LogNormal::fit(&latencies).unwrap();
let p99      = dist.inverse_cdf(0.99).unwrap();      // latency budget
let p_beyond = dist.sf(250.0).unwrap();              // P(X > 250 ms), exact tail
println!("p99 = {p99:.1} ms, P(>250ms) = {p_beyond:.2e}");
```

**"Simulate from it."** Every distribution samples through the same trait:

```rust
use rs_stats::Distribution;
use rs_stats::distributions::normal_distribution::Normal;
use rand::SeedableRng;

let d = Normal::new(100.0, 15.0).unwrap();
let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42); // reproducible
let draws = d.sample_n(&mut rng, 10_000).unwrap();
```

**"Did the change actually help?"** Parametric or not, with one-sided alternatives and effect sizes:

```rust
use rs_stats::hypothesis_tests::{
    Alternative, mann_whitney_u, two_sample_t_test_alt, cohens_d,
};

let before = [212.0, 198.5, 205.1, 220.8, 199.9, 210.4, 215.2, 202.7];
let after  = [188.2, 179.4, 195.0, 183.6, 176.9, 190.1, 181.3, 186.5];

// Welch's t-test, H₁: "after" is lower.
let t = two_sample_t_test_alt(&after, &before, false, Alternative::Less).unwrap();
// Distribution-free cross-check.
let u = mann_whitney_u(&after, &before, Alternative::Less).unwrap();
let d = cohens_d(&before, &after).unwrap();

println!("t-test p = {:.2e}, Mann-Whitney p = {:.2e}, Cohen's d = {d:.2}",
         t.p_value, u.p_value);
let (lo, hi) = t.confidence_interval(0.95).unwrap();
println!("95% CI of the difference: [{lo:.1}, {hi:.1}] ms");
```

---

## Why this library

**Tails you can quote.** `sf(x)` (survival function) has an exact closed form on all 14 distributions — `Normal.sf(10.0)` returns `7.6e-24`, not `0.0`. p-values, `erfc`, and the incomplete gamma/beta tails keep full *relative* precision where `1.0 - cdf(x)` collapses. Quantile brackets expand dynamically, so heavy-tailed quantiles (Student-t with ν < 2, F with small denominators) are correct instead of silently clamped.

**Cross-validated against scipy/numpy.** Every distribution (pdf, cdf, sf, quantiles, log-likelihoods), every test statistic and p-value, every regression output is checked against scipy/numpy references — on friendly inputs *and* hostile ones: quantiles at 10⁻⁶, n = 5000 binomials, u64-edge combinatorics, ν = 0.5 Student-t.

**Fast where it matters.** Discrete CDFs are O(1) closed forms (regularized incomplete gamma/beta). `fit_all` sorts once, shares it across all KS tests, and computes each log-likelihood exactly once. Decision-tree split search is a single incremental sweep. Rayon parallelism kicks in only past size thresholds where it actually wins.

**Panic-free, feature-gated.** All fallible operations return `StatsResult<T>`. Degenerate inputs (zero variance, NaN parameters, all-zero tables) are typed errors, not NaN propagation. `--no-default-features` gives you the pure math with just `num-traits` and `rand`.

---

## Tour

### Distributions

| Continuous | Discrete | Multivariate |
|---|---|---|
| Normal, LogNormal, Exponential, Uniform, Gamma, Weibull, Beta, ChiSquared, StudentT, F, Cauchy, Laplace, Pareto, Logistic | Poisson, Binomial, Geometric, NegativeBinomial | MultivariateNormal (Cholesky-backed pdf/sample/Mahalanobis) |

Each implements the unified `Distribution` trait:

```rust
use rs_stats::Distribution;
use rs_stats::distributions::gamma_distribution::Gamma;

let g = Gamma::fit(&data)?;             // MLE / method-of-moments
g.pdf(x)?;      g.logpdf(x)?;           // density (log-space stable)
g.cdf(x)?;      g.sf(x)?;               // CDF and exact upper tail
g.inverse_cdf(0.999)?;                  // quantiles
g.sample(&mut rng)?;                    // random draws
g.mean();  g.variance();  g.aic(&data)?;  g.bic(&data)?;
```

The trait is object-safe: `Box<dyn Distribution<X = f64>>` works for runtime polymorphism (`X = u64` for the discrete family).

### Automatic fitting

```rust
use rs_stats::{fit_all, fit_all_verbose};

let ranked = fit_all(&data)?;                  // all candidates, sorted by AIC
let (fits, skipped) = fit_all_verbose(&data)?; // + why each candidate failed
for f in &ranked {
    println!("{:<12} AIC={:>8.1}  KS p={:.3}", f.name, f.aic, f.ks_p_value);
}
```

### Hypothesis tests

| Parametric | Non-parametric & resampling | Categorical & meta |
|---|---|---|
| one-sample / two-sample (Student & Welch) / paired t-tests, one-way ANOVA (+ η²), D'Agostino K² normality | Mann-Whitney U, Wilcoxon signed-rank, two-sample Kolmogorov-Smirnov, bootstrap CIs, permutation tests | χ² goodness-of-fit & independence, Fisher exact (2×2), p-value adjustment (Bonferroni / Holm / BH) |

Plus **exact power analysis** through the noncentral t distribution:

```rust
use rs_stats::hypothesis_tests::{sample_size_t_test, power_t_test, Alternative, TTestKind};

// "How many subjects per arm to detect d = 0.5 at 80% power?"  → 64
let n = sample_size_t_test(TTestKind::TwoSample, 0.5, 0.05, 0.8, Alternative::TwoSided)?;
```

And distribution-free inference for **any** statistic:

```rust
use rs_stats::resampling::{bootstrap_ci, permutation_test};
use rs_stats::prob::quantile;

// 95% CI for the p90 latency — no closed form needed.
let ci = bootstrap_ci(&latencies, |s| quantile(s, 0.9).unwrap(), 5000, 0.95, &mut rng)?;
```

All two-sample tests take `Alternative::{TwoSided, Less, Greater}`; t-test results expose `confidence_interval(level)`; small tables get the exact test:

```rust
use rs_stats::hypothesis_tests::{fisher_exact, Alternative};

// Responders: treatment 8/10, control 1/6 — too small for χ².
let r = fisher_exact(8, 2, 1, 5, Alternative::TwoSided)?;
println!("odds ratio = {:.0}, exact p = {:.4}", r.odds_ratio, r.p_value);
```

### Correlation & descriptive statistics

```rust
use rs_stats::prob::{pearson_test, spearman, describe, quantile};

let r = pearson_test(&x, &y)?;          // r, t-statistic, p-value
let rho = spearman(&x, &y)?;            // rank correlation, tie-aware

let d = describe(&data)?;               // n, mean, std-dev, min/Q1/median/Q3/max
let p95 = quantile(&data, 0.95)?;       // numpy-compatible interpolation
```

### Regression

```rust
use rs_stats::regression::linear_regression::LinearRegression;
use rs_stats::regression::multiple_linear_regression::MultipleLinearRegression;

let mut lr = LinearRegression::<f64>::new();
lr.fit(&x, &y)?;
let (lo, hi) = lr.confidence_interval(x0, 0.95)?;   // CI of the mean response
let (plo, phi) = lr.prediction_interval(x0, 0.95)?; // bounds for one new point

let mut mlr = MultipleLinearRegression::<f64>::new();
mlr.fit(&rows, &y)?;
// Per-coefficient inference, like a real stats package:
for (i, (se, p)) in mlr.coefficient_std_errors.iter().zip(&mlr.p_values).enumerate() {
    println!("β{i} = {:.3} (SE {:.3}, p = {:.4})", mlr.coefficients[i], se, p);
}
println!("F = {:.1} (p = {:.2e})", mlr.f_statistic, mlr.f_p_value);
```

Decision trees (CART) handle regression **and** classification with plain `f64` targets — `DecisionTree<f64, f64>` — using incremental split search (running sums for MSE, class-count arrays for Gini/entropy) and a median-based MAE criterion:

```rust
use rs_stats::regression::decision_tree::{DecisionTree, SplitCriterion, TreeType};

let mut tree = DecisionTree::<f64, f64>::new(TreeType::Regression, SplitCriterion::Mse, 8, 4, 2);
tree.fit(&features, &targets)?;
let preds = tree.predict(&new_features)?;
let importances = tree.feature_importances();
```

### Streaming statistics

Welford online estimators for data that doesn't fit in memory (or arrives one point at a time): scalar, per-axis vector, and full covariance-matrix variants, all with O(1)/O(D²) updates, `merge()` for parallel reduction, and zero steady-state allocation.

```rust
use rs_stats::prob::welford::Welford;

let mut w = Welford::new();
for x in stream { w.push(x); }
println!("mean = {}, sample var = {}", w.mean(), w.variance_sample()?);
```

---

## Cargo features

| Feature | Default | What it adds |
|---|---|---|
| `parallel` | ✅ | Rayon-backed parallelism (`fit_all`, decision trees, ANOVA, bulk predict) with sequential fallbacks below size thresholds |
| `serde` | ✅ | Serde derives on distributions and models + `save`/`load`/`to_json` persistence |

```toml
# Minimal build: just the math, two dependencies (num-traits, rand).
rs-stats = { version = "4", default-features = false }
```

## Performance notes

Measured with criterion on the v4.0 refactors (vs v3.0):

| Path | Change |
|---|---|
| `erf` / `erfc` / `Normal::cdf` | iterative incomplete gamma → Cody rational approximations: **−69% to −88%** (~2–6 ns/call, ~1 ulp) |
| `Poisson::cdf`, `NegativeBinomial::cdf` | O(k) sums → **O(1)** closed forms (−99.5% at k = 1000) |
| `fit_all` | one shared sort + single log-likelihood pass: **−24%** (n = 10⁴) to **−75%** (n = 50) |
| `StudentT::log_likelihood` (n = 10⁴) | normalization constants hoisted: **−90%** |
| Decision-tree fit | incremental split sweep: **4–25× faster**, more on larger nodes |

Hot reduction loops use multiple independent accumulators so they pipeline
and auto-vectorise without `-ffast-math`. For an extra free boost in *your*
binary, allow the compiler to use your CPU's full instruction set
(AVX2/FMA/NEON):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Error handling

Everything fallible returns `StatsResult<T> = Result<T, StatsError>`, with typed variants (`InvalidInput`, `InvalidParameter`, `DimensionMismatch`, `EmptyData`, `ConversionError`, `NumericalError`, `NotFitted`, …). Degenerate inputs — zero-variance t-tests, all-zero χ² tables, `z_score` with σ = 0, non-finite distribution parameters — are **errors**, never silent NaN or ±∞.

```rust
use rs_stats::distributions::normal_distribution::Normal;

match Normal::new(0.0, f64::NAN) {
    Err(e) => eprintln!("caught: {e}"), // "Normal::new: parameters must be finite"
    Ok(_) => unreachable!(),
}
```

## What's new in 4.0

- 4 new distributions (Cauchy, Laplace, Pareto, Logistic — 14 auto-fit
  candidates) + MultivariateNormal with Cholesky sampling
- Bootstrap confidence intervals & permutation tests for any statistic
- Exact power / sample-size analysis (noncentral t), D'Agostino K²
  normality test, multiple-comparison corrections (Bonferroni/Holm/BH)
- Marsaglia-Tsang gamma sampling — Gamma, Beta, χ², Student-t and F all
  sample without quantile bisection; ziggurat for Normal/LogNormal
- `sample()` / `sample_n()` and exact `sf()` on all distributions
- Pearson / Spearman correlation with significance tests
- Mann-Whitney U, Wilcoxon signed-rank, two-sample KS, Fisher exact
- One-sided alternatives + confidence intervals on t-tests; Cohen's d, η²
- Per-coefficient SE / t / p and global F-test in multiple regression
- Real Student-t confidence *and* prediction intervals in linear regression
- `describe()` / `quantile()` descriptive helpers
- Decision trees with `f64` targets, incremental splits, median MAE
- Deep-tail correctness fixes across quantiles, `erfc`, large-n binomials
- Cody rational `erf`/`erfc` (~1 ulp, up to 8× faster `Normal::cdf`)
- Cargo features `parallel` / `serde`; leaner dependency tree

Breaking changes: χ² tests now return `ChiSquareResult` (named fields), `std_err` uses the sample convention (`std_err_population` keeps the old one), Exponential's out-of-support behaviour is aligned with the other distributions, and degenerate inputs that used to return NaN/∞ now return errors.

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feat/my-feature`
3. Commit: `git commit -m "feat(scope): description"`
4. Push and open a pull request

All PRs must pass `cargo test`, `cargo clippy -- -D warnings`, and `cargo fmt --check`.

## License

MIT — see [LICENSE](LICENSE).
