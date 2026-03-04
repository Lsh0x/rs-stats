# rs-stats

[![Rust](https://img.shields.io/badge/rust-1.56%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.2-green.svg)](https://crates.io/crates/rs-stats)
[![Tests](https://img.shields.io/badge/tests-343%20passing-brightgreen.svg)](https://github.com/lsh0x/rs-stats/actions)
[![CI](https://github.com/lsh0x/rs-stats/workflows/CI/badge.svg)](https://github.com/lsh0x/rs-stats/actions)
[![Docs](https://docs.rs/rs-stats/badge.svg)](https://docs.rs/rs-stats)
[![Crates.io](https://img.shields.io/crates/v/rs-stats.svg)](https://crates.io/crates/rs-stats)

A comprehensive statistical library written in Rust, designed for data scientists, researchers and engineers who need reliable, production-grade statistics.

**rs-stats** covers the full statistical pipeline: probability functions, 14 parametric distributions with MLE/MOM fitting, automatic distribution detection, Kolmogorov-Smirnov goodness-of-fit tests, hypothesis testing, and regression analysis — all **panic-free** via `StatsResult<T>`.

---

## Table of Contents

- [Key Features](#-key-features)
- [Installation](#installation)
- [Quick Start — Medical Example](#quick-start--medical-example)
- [Distributions](#distributions)
  - [Continuous Distributions](#continuous-distributions)
  - [Discrete Distributions](#discrete-distributions)
- [Automatic Distribution Fitting](#automatic-distribution-fitting)
- [Hypothesis Testing](#hypothesis-testing)
- [Regression Analysis](#regression-analysis)
- [Error Handling](#error-handling)
- [Documentation](#documentation)

---

## ✨ Key Features

- **14 parametric distributions** — continuous and discrete, each with `fit()` (MLE or MOM), PDF/PMF, CDF, quantile, mean, variance, AIC, BIC
- **Unified trait interface** — `Distribution` and `DiscreteDistribution` traits enable polymorphism and `Box<dyn Distribution>` at runtime
- **Auto-fit API** — detect data type, fit all candidates, rank by AIC/BIC/KS-test in a single call
- **Kolmogorov-Smirnov goodness-of-fit** — continuous and discrete variants
- **Special functions** — `ln_gamma`, regularized incomplete gamma and beta (Lanczos + Numerical Recipes)
- **Hypothesis testing** — t-tests (one-sample, two-sample, paired), ANOVA, chi-square, chi-square independence
- **Regression** — linear, multiple linear, decision trees (regression and classification)
- **Panic-free** — every computation returns `StatsResult<T>`, ready for production

---

## Installation

```toml
[dependencies]
rs-stats = "2.0.2"
```

Or:

```bash
cargo add rs-stats
```

---

## Quick Start — Medical Example

> **Scenario**: You receive anonymised blood pressure measurements from 1 200 patients in a hypertension study. You want to identify the best-fitting distribution, compute the probability of a dangerously high reading, and compare two treatment arms.

```rust
use rs_stats::{auto_fit, fit_all, Distribution};
use rs_stats::distributions::normal_distribution::Normal;
use rs_stats::hypothesis_tests::t_test::two_sample_t_test;

// ── Step 1: systolic blood pressure data (mmHg) ──────────────────────────────
// Real study would have 1 200 values; small sample used for illustration
let systolic_bp = vec![
    115.0, 122.0, 118.0, 130.0, 125.0, 119.0, 128.0, 132.0,
    121.0, 117.0, 126.0, 135.0, 123.0, 120.0, 127.0, 131.0,
];

// ── Step 2: auto-detect the distribution and find the best fit ────────────────
let best = auto_fit(&systolic_bp)?;
println!("Best distribution : {}", best.name);   // → Normal
println!("  AIC             : {:.2}", best.aic);
println!("  KS p-value      : {:.4}", best.ks_p_value);

// ── Step 3: use the fitted Normal to answer clinical questions ────────────────
let bp_dist = Normal::fit(&systolic_bp)?;
println!("Fitted Normal(μ={:.1}, σ={:.1})", bp_dist.mean(), bp_dist.std_dev());

// P(BP > 140 mmHg) — hypertensive threshold
let p_hyper = 1.0 - bp_dist.cdf(140.0)?;
println!("P(BP > 140 mmHg) = {:.2}%", p_hyper * 100.0);

// 95th percentile — what value do 95% of patients fall below?
let p95 = bp_dist.inverse_cdf(0.95)?;
println!("95th percentile  = {:.1} mmHg", p95);

// ── Step 4: compare two treatment arms ───────────────────────────────────────
let control_arm   = vec![128.0, 132.0, 125.0, 130.0, 129.0, 131.0, 127.0, 133.0];
let treatment_arm = vec![118.0, 122.0, 115.0, 120.0, 119.0, 121.0, 117.0, 123.0];

let t_result = two_sample_t_test(&control_arm, &treatment_arm, false)?;
println!("Two-sample t-test: t={:.3}, p={:.4}", t_result.t_statistic, t_result.p_value);
if t_result.p_value < 0.05 {
    println!("→ Statistically significant BP reduction (α = 0.05)");
}
```

---

## Distributions

### Continuous Distributions

All continuous distributions implement the `Distribution` trait and expose:
- `Dist::new(params)` — validated constructor
- `Dist::fit(data)` — maximum likelihood (MLE) or method-of-moments (MOM) estimation
- `.pdf(x)`, `.logpdf(x)`, `.cdf(x)`, `.inverse_cdf(p)` — core functions
- `.mean()`, `.variance()`, `.std_dev()` — moments
- `.aic(data)`, `.bic(data)` — model selection criteria

---

#### Normal — `distributions::normal_distribution::Normal`

> **When to use**: Symmetric continuous measurements that cluster around a mean.
> **Medical examples**: Blood pressure, height, weight in large populations, IQ scores, measurement errors in lab instruments.

```rust
use rs_stats::distributions::normal_distribution::Normal;
use rs_stats::Distribution;

// Diastolic blood pressure in a healthy cohort: N(80, 8)
let bp = Normal::new(80.0, 8.0)?;

// P(diastolic BP > 90 mmHg) — stage 1 hypertension threshold
let p_high = 1.0 - bp.cdf(90.0)?;
println!("P(DBP > 90) = {:.1}%", p_high * 100.0);   // ≈ 10.6%

// 97.5th percentile — upper reference range
let upper_ref = bp.inverse_cdf(0.975)?;
println!("Upper reference (97.5th pct) = {:.1} mmHg", upper_ref);  // ≈ 95.7

// Fit to patient data (MLE: μ̂ = mean, σ̂ = pop std-dev)
let measurements = vec![78.0, 82.0, 79.0, 85.0, 81.0, 77.0, 83.0, 80.0];
let fitted = Normal::fit(&measurements)?;
println!("Fitted μ = {:.2}, σ = {:.2}", fitted.mean(), fitted.std_dev());
```

---

#### Log-Normal — `distributions::lognormal::LogNormal`

> **When to use**: Right-skewed positive data — concentrations, durations, biological assays.
> **Medical examples**: CRP (C-reactive protein) levels, serum creatinine, drug plasma concentrations, tumour volumes, hospital length-of-stay.

```rust
use rs_stats::distributions::lognormal::LogNormal;
use rs_stats::Distribution;

// CRP levels (mg/L) in an outpatient cohort
// CRP is log-normally distributed: healthy < 5, elevated 5–100, critical > 100
let crp_data = vec![
    1.2, 0.8, 2.1, 1.5, 45.0, 3.2, 0.9, 12.4, 1.8, 88.0,
    2.4, 1.1, 5.6, 0.7, 22.3, 3.9, 1.3, 9.7,  0.6,  0.5,
];

let crp = LogNormal::fit(&crp_data)?;
println!("LogNormal(μ={:.2}, σ={:.2})", crp.mu, crp.sigma);

// Median CRP (more informative than mean for skewed data)
let median = crp.inverse_cdf(0.5)?;
println!("Median CRP     = {:.2} mg/L", median);

// P(CRP > 10 mg/L) — significant inflammation threshold
let p_inflamed = 1.0 - crp.cdf(10.0)?;
println!("P(CRP > 10)    = {:.1}%", p_inflamed * 100.0);
```

---

#### Weibull — `distributions::weibull::Weibull`

> **When to use**: Time-to-event data where the hazard rate changes over time.
> **Medical examples**: Time to relapse after cancer treatment, medical device/implant survival, time until a drug loses efficacy, organ transplant survival.

```rust
use rs_stats::distributions::weibull::Weibull;
use rs_stats::Distribution;

// Time to relapse (months) after chemotherapy — k > 1 means increasing hazard
let relapse_times = vec![3.1, 7.4, 12.5, 2.8, 18.2, 5.9, 9.6, 15.3, 4.2, 22.0];

let w = Weibull::fit(&relapse_times)?;
println!("Weibull(k={:.2}, λ={:.2})", w.k, w.lambda);
// k > 1 → hazard rate increases over time (survivors become more at risk)

// Median relapse-free survival
let median_rfs = w.inverse_cdf(0.5)?;
println!("Median relapse-free survival = {:.1} months", median_rfs);

// P(relapse within 6 months) — short-term risk
let p_6mo = w.cdf(6.0)?;
println!("P(relapse < 6 months)        = {:.1}%", p_6mo * 100.0);

// 1-year survival probability
let p_1yr = 1.0 - w.cdf(12.0)?;
println!("1-year relapse-free survival = {:.1}%", p_1yr * 100.0);
```

---

#### Gamma — `distributions::gamma_distribution::Gamma`

> **When to use**: Positive right-skewed data, especially waiting times or accumulated effects.
> **Medical examples**: ICU length-of-stay, time between hospital readmissions, blood glucose AUC in OGTT.

```rust
use rs_stats::distributions::gamma_distribution::Gamma;
use rs_stats::Distribution;

// ICU length-of-stay (days) — Gamma naturally models positive skewed durations
let icu_los = vec![
    1.5, 2.0, 4.5, 1.2, 7.8, 3.1, 2.4, 10.2, 1.8, 5.6,
    3.9, 2.1, 6.3, 1.4, 8.9, 4.0, 2.7,  3.5, 1.9, 12.1,
];

let gamma = Gamma::fit(&icu_los)?;
println!("Gamma(α={:.2}, β={:.2})", gamma.alpha, gamma.beta);
println!("Mean ICU stay   = {:.2} days", gamma.mean());
println!("Std-dev         = {:.2} days", gamma.std_dev());

// P(LOS > 7 days) — prolonged ICU stay threshold for resource planning
let p_prolonged = 1.0 - gamma.cdf(7.0)?;
println!("P(LOS > 7 days) = {:.1}%", p_prolonged * 100.0);
```

---

#### Beta — `distributions::beta::Beta`

> **When to use**: Proportions, rates, and probabilities bounded in (0, 1).
> **Medical examples**: Diagnostic test sensitivity and specificity, medication adherence rates, tumour response rates, proportion of time in therapeutic range (TTR) for anticoagulant patients.

```rust
use rs_stats::distributions::beta::Beta;
use rs_stats::Distribution;

// Time-in-therapeutic-range (TTR) for warfarin patients (values in 0–1)
// TTR ≥ 0.70 is considered well-controlled anticoagulation
let ttr_data = vec![
    0.72, 0.65, 0.88, 0.55, 0.91, 0.78, 0.62, 0.84,
    0.70, 0.58, 0.79, 0.93, 0.67, 0.75, 0.48, 0.82,
];

let beta = Beta::fit(&ttr_data)?;
println!("Beta(α={:.2}, β={:.2})", beta.alpha, beta.beta);

// P(TTR ≥ 0.70) — probability of being well-controlled
let p_well = 1.0 - beta.cdf(0.70)?;
println!("P(TTR ≥ 0.70) = {:.1}%", p_well * 100.0);

// Median TTR across the population
let median_ttr = beta.inverse_cdf(0.5)?;
println!("Median TTR    = {:.1}%", median_ttr * 100.0);
```

---

#### Student's t — `distributions::student_t::StudentT`

> **When to use**: Symmetric distributions with heavier tails than Normal; small-sample inference.
> **Medical examples**: Standardised effect sizes in small pilot studies, residuals from mixed-effects models, computing critical values for paired-samples tests.

```rust
use rs_stats::distributions::student_t::StudentT;
use rs_stats::Distribution;

// Small diabetes pilot study (n=12): t-distribution with df = n-1 = 11
let t_dist = StudentT::new(0.0, 1.0, 11.0)?;

// Two-sided critical value at α = 0.05
let t_crit = t_dist.inverse_cdf(0.975)?;
println!("t-critical (α=0.05, df=11) = {:.3}", t_crit);   // ≈ 2.201

// p-value for an observed t-statistic of 2.5 (two-tailed)
let p_value = 2.0 * (1.0 - t_dist.cdf(2.5)?);
println!("p-value for |t|=2.5        = {:.4}", p_value);
```

---

#### Exponential — `distributions::exponential_distribution::Exponential`

> **When to use**: Time between events when events occur at a constant rate (memoryless property).
> **Medical examples**: Inter-arrival times in an emergency department, time between seizures in epilepsy patients, spontaneous adverse events during a trial.

```rust
use rs_stats::distributions::exponential_distribution::Exponential;
use rs_stats::Distribution;

// Time (minutes) between patient arrivals in an ED
let inter_arrivals = vec![8.2, 12.5, 4.1, 9.8, 6.3, 15.0, 3.7, 11.2, 7.4, 9.1];

let exp = Exponential::fit(&inter_arrivals)?;
println!("Exponential(λ={:.3} arrivals/min)", exp.lambda);
println!("Mean inter-arrival = {:.1} min", exp.mean());

// P(next patient within 5 minutes) — triage planning
let p_5min = exp.cdf(5.0)?;
println!("P(arrival < 5 min) = {:.1}%", p_5min * 100.0);
```

---

#### Chi-Squared — `distributions::chi_squared::ChiSquared`

> **When to use**: Distribution of sums of squared standard normals; used in goodness-of-fit tests and variance confidence intervals.
> **Medical examples**: Testing whether observed disease frequencies match expected proportions, variance confidence intervals for measurement devices.

```rust
use rs_stats::distributions::chi_squared::ChiSquared;
use rs_stats::Distribution;

// 6-category goodness-of-fit test: df = 6 - 1 = 5
let chi2 = ChiSquared::new(5.0)?;

// Critical value at α = 0.05
let chi2_crit = chi2.inverse_cdf(0.95)?;
println!("χ²(5) critical value (α=0.05) = {:.3}", chi2_crit);  // ≈ 11.07

// p-value for an observed χ² = 9.2
let p_value = 1.0 - chi2.cdf(9.2)?;
println!("p-value for χ²=9.2            = {:.4}", p_value);
```

---

#### F-Distribution — `distributions::f_distribution::FDistribution`

> **When to use**: Ratio of two chi-squared variables; used in ANOVA and regression significance tests.
> **Medical examples**: Comparing biomarker variance across patient groups, multi-arm ANOVA F-statistic, F-test in multiple regression predicting clinical outcomes.

```rust
use rs_stats::distributions::f_distribution::FDistribution;
use rs_stats::Distribution;

// ANOVA with 4 groups, total n=52: F(3, 48)
let f_dist = FDistribution::new(3.0, 48.0)?;

// Critical value at α = 0.05
let f_crit = f_dist.inverse_cdf(0.95)?;
println!("F(3,48) critical value (α=0.05) = {:.3}", f_crit);  // ≈ 2.80

// p-value for observed F = 4.5
let p_value = 1.0 - f_dist.cdf(4.5)?;
println!("p-value for F=4.5               = {:.4}", p_value);
```

---

#### Uniform — `distributions::uniform_distribution::Uniform`

> **When to use**: All values in a range are equally likely.
> **Medical examples**: Randomisation checks in clinical trials, uncertainty about a drug's effective window, boundary-condition stress testing.

```rust
use rs_stats::distributions::uniform_distribution::Uniform;
use rs_stats::Distribution;

// Drug release window: effective between 2 h and 6 h post-ingestion
let release = Uniform::new(2.0, 6.0)?;

// P(effective within the first 3 hours)
let p_3h = release.cdf(3.0)?;
println!("P(effective by 3h) = {:.1}%", p_3h * 100.0);  // 25%
```

---

### Discrete Distributions

All discrete distributions implement the `DiscreteDistribution` trait and expose:
- `Dist::new(params)` — validated constructor
- `Dist::fit(data)` — MLE or MOM from `&[f64]`
- `.pmf(k)`, `.logpmf(k)`, `.cdf(k)` — core functions
- `.mean()`, `.variance()`, `.std_dev()` — moments
- `.aic(data)`, `.bic(data)` — model selection

---

#### Poisson — `distributions::poisson_distribution::Poisson`

> **When to use**: Count of rare independent events in a fixed time or space window.
> **Medical examples**: Adverse drug reactions per 1 000 prescriptions, surgical site infections per month, emergency calls per hour, mutations per cell division.

```rust
use rs_stats::distributions::poisson_distribution::Poisson;
use rs_stats::DiscreteDistribution;

// Hospital-acquired infections (HAI) per ward per month: λ = 2.3
let hai = Poisson::new(2.3)?;

println!("P(0 HAI)     = {:.1}%", hai.pmf(0)? * 100.0);   // ≈ 10.0%
println!("P(≥5 HAI)    = {:.1}%", (1.0 - hai.cdf(4)?) * 100.0);  // alert threshold

// Fit from 12 months of observed counts
let monthly_counts = vec![1.0, 3.0, 2.0, 0.0, 4.0, 2.0, 1.0, 3.0, 2.0, 1.0, 5.0, 2.0];
let fitted = Poisson::fit(&monthly_counts)?;
println!("Estimated λ  = {:.2} infections/month", fitted.lambda);
```

---

#### Binomial — `distributions::binomial_distribution::Binomial`

> **When to use**: Number of successes in n independent trials with constant probability p.
> **Medical examples**: Responders in a treatment cohort, positive tests in a screening batch, side-effect events in a treated group.

```rust
use rs_stats::distributions::binomial_distribution::Binomial;
use rs_stats::DiscreteDistribution;

// Trial: n=100 patients, literature response rate p=0.35
let trial = Binomial::new(100, 0.35)?;

println!("E[responders]   = {:.0}", trial.mean());   // 35

// P(≥ 45 responders) — detect a meaningful improvement
let p_improved = 1.0 - trial.cdf(44)?;
println!("P(≥45 respond)  = {:.2}%", p_improved * 100.0);
```

---

#### Geometric — `distributions::geometric::Geometric`

> **When to use**: Number of trials until the first success (k ≥ 1).
> **Medical examples**: Screening cycles until a lesion is detected, treatment attempts until remission, needle passes until a successful lumbar puncture.

```rust
use rs_stats::distributions::geometric::Geometric;
use rs_stats::DiscreteDistribution;

// Colonoscopy screening: P(detecting polyp per session) = 0.18
let screening = Geometric::new(0.18)?;

println!("E[sessions to detect] = {:.1}", screening.mean());   // ≈ 5.6
let p_within_3 = screening.cdf(3)?;
println!("P(detected ≤ 3 sessions) = {:.1}%", p_within_3 * 100.0);
```

---

#### Negative Binomial — `distributions::negative_binomial::NegativeBinomial`

> **When to use**: Overdispersed count data (variance > mean), or number of failures before r-th success.
> **Medical examples**: Hospitalisations before stable remission, overdispersed adverse event counts, recurrences before sustained response.

```rust
use rs_stats::distributions::negative_binomial::NegativeBinomial;
use rs_stats::DiscreteDistribution;

// Re-admissions before stable remission — overdispersed (variance > mean)
let admissions = vec![
    0.0, 2.0, 1.0, 5.0, 3.0, 0.0, 4.0, 1.0, 2.0, 6.0,
    1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 0.0, 2.0, 3.0, 1.0,
];

let nb = NegativeBinomial::fit(&admissions)?;
println!("NegBin(r={:.2}, p={:.3})", nb.r, nb.p);
println!("Mean re-admissions = {:.2}", nb.mean());
println!("P(0 re-admissions) = {:.1}%", nb.pmf(0)? * 100.0);
```

---

## Automatic Distribution Fitting

> **Scenario**: A pharmacokineticist wants to know which distribution best describes drug half-life across 80 patients, without assuming Normality.

```rust
use rs_stats::{auto_fit, fit_all};

// Drug half-life (hours) — typically log-normal or Weibull in PK studies
let half_lives = vec![
    4.2, 6.1, 3.8, 9.5, 5.3, 7.4, 4.9, 11.2, 3.5, 6.8,
    8.1, 4.4, 5.7, 7.0, 3.9, 10.3, 5.1,  6.5, 4.7,  8.6,
];

// One-call: auto-detect type + best AIC
let best = auto_fit(&half_lives)?;
println!("Best fit: {} (AIC={:.2}, KS p={:.3})", best.name, best.aic, best.ks_p_value);

// Full ranking — compare all candidates
println!("\n{:<15} {:>8} {:>8} {:>10}", "Distribution", "AIC", "BIC", "KS p-value");
println!("{}", "-".repeat(45));
for r in fit_all(&half_lives)? {
    println!("{:<15} {:>8.2} {:>8.2} {:>10.4}", r.name, r.aic, r.bic, r.ks_p_value);
}
// Typical output:
// Distribution    AIC      BIC   KS p-value
// -----------------------------------------
// LogNormal     82.34    84.12     0.8231
// Gamma         83.71    85.49     0.7654
// Weibull       84.02    85.80     0.7412
// Normal        89.45    91.23     0.4103
```

### Available candidates

| Type | Distributions |
|------|--------------|
| Continuous (`fit_all`) | Normal, Exponential, Uniform, Gamma, LogNormal, Weibull, Beta, StudentT, F, ChiSquared |
| Discrete (`fit_all_discrete`) | Poisson, Geometric, NegativeBinomial, Binomial |

---

## Hypothesis Testing

> **Scenario**: A clinical trial compares HbA1c reduction across three diabetes treatments.

```rust
use rs_stats::hypothesis_tests::{
    t_test::{one_sample_t_test, two_sample_t_test, paired_t_test},
    anova::one_way_anova,
    chi_square_test::chi_square_independence,
};

// ── Paired t-test: before vs after treatment ──────────────────────────────────
let before = vec![8.2, 7.9, 8.6, 9.1, 8.4, 8.0, 8.8, 9.3];  // HbA1c %
let after  = vec![7.4, 7.1, 7.9, 8.3, 7.5, 7.2, 8.0, 8.5];
let paired = paired_t_test(&before, &after)?;
println!("Paired t-test: t={:.3}, p={:.4}", paired.t_statistic, paired.p_value);
// p < 0.05 → significant reduction in HbA1c

// ── One-way ANOVA: compare three treatment arms ───────────────────────────────
let drug_a = vec![-0.8, -1.2, -0.5, -1.5, -0.9];  // HbA1c change %
let drug_b = vec![-1.4, -1.8, -1.1, -2.0, -1.6];
let drug_c = vec![-0.4, -0.6, -0.3, -0.8, -0.5];
let groups: Vec<&[f64]> = vec![&drug_a, &drug_b, &drug_c];
let anova = one_way_anova(&groups)?;
println!("ANOVA: F={:.3}, p={:.4}", anova.f_statistic, anova.p_value);

// ── Chi-square independence: side-effect rate by treatment ────────────────────
// Rows: Drug A, B, C  |  Cols: No side-effect, Side-effect occurred
let observed = vec![
    vec![42, 8],   // Drug A
    vec![36, 14],  // Drug B
    vec![45, 5],   // Drug C
];
let (_chi2, _df, p) = chi_square_independence(&observed)?;
println!("χ² independence: p={:.4}", p);
```

---

## Regression Analysis

> **Scenario**: Predict post-operative recovery time from patient characteristics.

```rust
use rs_stats::regression::linear_regression::LinearRegression;
use rs_stats::regression::multiple_linear_regression::MultipleLinearRegression;

// ── Simple linear regression: age → recovery time ────────────────────────────
let age           = vec![35.0, 45.0, 55.0, 62.0, 70.0, 48.0, 58.0, 40.0];
let recovery_days = vec![ 4.0,  5.5,  7.0,  8.5, 10.0,  6.0,  7.5,  5.0];

let mut model = LinearRegression::new();
model.fit(&age, &recovery_days)?;
println!("Recovery ~ Age: slope={:.3} d/yr, R²={:.4}", model.slope, model.r_squared);

// Predict for a 52-year-old patient with 95% CI
let predicted = model.predict(52.0);
let (lo, hi)  = model.confidence_interval(52.0, 0.95)?;
println!("Predicted (age=52): {:.1} days  95% CI [{:.1}, {:.1}]", predicted, lo, hi);

// ── Multiple regression: age + BMI + comorbidity score → recovery ─────────────
let features = vec![
    vec![35.0, 23.0, 1.0],   // [age, BMI, comorbidity score]
    vec![55.0, 28.0, 2.0],
    vec![62.0, 31.0, 3.0],
    vec![45.0, 25.0, 1.0],
    vec![70.0, 33.0, 4.0],
    vec![48.0, 26.0, 2.0],
];
let outcomes = vec![4.5, 7.2, 9.8, 5.1, 12.3, 6.4];

let mut mlr = MultipleLinearRegression::new();
mlr.fit(&features, &outcomes)?;
println!("MLR R² = {:.4}, Adj. R² = {:.4}", mlr.r_squared, mlr.adjusted_r_squared);
```

---

## Error Handling

All functions return `StatsResult<T>` — a type alias for `Result<T, StatsError>`. The library **never panics**.

```rust
use rs_stats::{StatsError, StatsResult};
use rs_stats::distributions::normal_distribution::Normal;
use rs_stats::Distribution;

fn reference_range(mean: f64, sd: f64) -> StatsResult<(f64, f64)> {
    let dist  = Normal::new(mean, sd)?;          // Err if sd ≤ 0
    let lower = dist.inverse_cdf(0.025)?;
    let upper = dist.inverse_cdf(0.975)?;
    Ok((lower, upper))
}

match reference_range(80.0, -5.0) {             // Invalid: negative SD
    Ok((lo, hi)) => println!("Ref range: [{:.1}, {:.1}]", lo, hi),
    Err(StatsError::InvalidInput { message }) =>
        println!("Invalid params: {}", message), // → "Normal::new: std_dev must be positive"
    Err(e) => println!("Error: {}", e),
}
```

### Error Variants

| Variant | Raised when |
|---------|-------------|
| `InvalidInput` | Out-of-domain parameter (negative σ, p ∉ [0,1], …) |
| `EmptyData` | Empty slice passed to `fit()` or statistical functions |
| `DimensionMismatch` | Mismatched array lengths (regression, paired tests) |
| `ConversionError` | Type conversion failures |
| `NumericalError` | Numerical instability (overflow, NaN propagation) |
| `NotFitted` | `predict()` called before `fit()` on a regression model |

---

## Documentation

```bash
cargo doc --open          # full API documentation
cargo test                # 343 unit tests + 66 doc tests
cargo clippy              # linting
cargo fmt --check         # formatting
```

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feat/my-feature`
3. Commit: `git commit -m "feat(scope): description"`
4. Push and open a pull request

All PRs must pass `cargo test`, `cargo clippy -- -D warnings`, and `cargo fmt --check`.

---

## License

MIT — see [LICENSE](LICENSE).
