//! # Distribution Fitting
//!
//! High-level API for automatic distribution detection and fitting.
//!
//! Given a dataset, this module:
//! 1. **Detects** whether the data is discrete or continuous (`detect_data_type`)
//! 2. **Fits** all applicable distribution candidates using MLE or MOM
//! 3. **Ranks** them by AIC (lower = better fit, penalised for complexity)
//! 4. **Validates** with a Kolmogorov-Smirnov goodness-of-fit test
//!
//! ## Key functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | `auto_fit(data)` | Auto-detect type + return single best fit |
//! | `fit_all(data)` | All 10 continuous distributions, ranked by AIC |
//! | `fit_best(data)` | Best continuous distribution (lowest AIC) |
//! | `fit_all_discrete(data)` | All 4 discrete distributions, ranked by AIC |
//! | `fit_best_discrete(data)` | Best discrete distribution |
//! | `detect_data_type(data)` | `DataKind::Discrete` or `DataKind::Continuous` |
//! | `ks_test(data, cdf)` | Two-sided KS test for continuous distributions |
//! | `ks_test_discrete(data, cdf)` | KS test for discrete distributions |
//!
//! ## Medical example — identifying the best distribution for drug half-life
//!
//! ```rust
//! use rs_stats::distributions::fitting::{fit_all, auto_fit};
//!
//! // Drug half-life (hours) measured in 20 patients — typically log-normal in PK studies
//! let half_lives = vec![
//!     4.2, 6.1, 3.8, 9.5, 5.3, 7.4, 4.9, 11.2, 3.5, 6.8,
//!     8.1, 4.4, 5.7, 7.0, 3.9, 10.3, 5.1,  6.5, 4.7,  8.6,
//! ];
//!
//! // One-call: auto-detect + return single best (lowest AIC)
//! let best = auto_fit(&half_lives).unwrap();
//! println!("Best fit: {} (AIC={:.2}, KS p={:.3})", best.name, best.aic, best.ks_p_value);
//!
//! // Full ranking for model comparison and reporting
//! println!("{:<15} {:>8} {:>8} {:>10}", "Distribution", "AIC", "BIC", "KS p-value");
//! for r in fit_all(&half_lives).unwrap() {
//!     println!("{:<15} {:>8.2} {:>8.2} {:>10.4}", r.name, r.aic, r.bic, r.ks_p_value);
//! }
//! ```
//!
//! ## Medical example — discrete event counts (adverse reactions)
//!
//! ```rust
//! use rs_stats::distributions::fitting::fit_all_discrete;
//!
//! // Adverse drug reaction counts per patient over 6 months
//! let adr_counts = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0,
//!                       2.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 2.0, 1.0, 0.0];
//!
//! for r in fit_all_discrete(&adr_counts).unwrap() {
//!     println!("{:<20} AIC={:.2}  KS p={:.3}", r.name, r.aic, r.ks_p_value);
//! }
//! // Poisson usually wins when variance ≈ mean; NegativeBinomial wins if overdispersed
//! ```

use crate::distributions::{
    beta::Beta,
    binomial_distribution::Binomial,
    chi_squared::ChiSquared,
    f_distribution::FDistribution,
    gamma_distribution::Gamma,
    geometric::Geometric,
    lognormal::LogNormal,
    negative_binomial::NegativeBinomial,
    normal_distribution::Normal,
    poisson_distribution::Poisson,
    student_t::StudentT,
    traits::{DiscreteDistribution, Distribution},
    uniform_distribution::Uniform,
    weibull::Weibull,
};
use crate::error::{StatsError, StatsResult};

// ── Data kind detection ────────────────────────────────────────────────────────

/// Whether a dataset looks discrete or continuous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    /// All values are non-negative integers (whole numbers ≥ 0).
    Discrete,
    /// Contains non-integer or negative values — treated as continuous.
    Continuous,
}

/// Infer whether `data` is discrete (all non-negative integers) or continuous.
///
/// # Examples
/// ```
/// use rs_stats::distributions::fitting::{detect_data_type, DataKind};
///
/// assert_eq!(detect_data_type(&[0.0, 1.0, 2.0, 3.0]), DataKind::Discrete);
/// assert_eq!(detect_data_type(&[0.5, 1.5, 2.3]), DataKind::Continuous);
/// ```
pub fn detect_data_type(data: &[f64]) -> DataKind {
    if data
        .iter()
        .all(|&x| x >= 0.0 && x.fract() == 0.0 && x.is_finite())
    {
        DataKind::Discrete
    } else {
        DataKind::Continuous
    }
}

// ── Kolmogorov-Smirnov test ────────────────────────────────────────────────────

/// Result of a Kolmogorov-Smirnov goodness-of-fit test.
#[derive(Debug, Clone, Copy)]
pub struct KsResult {
    /// KS statistic D (maximum absolute deviation between empirical and theoretical CDF).
    pub statistic: f64,
    /// Approximate two-sided p-value.
    pub p_value: f64,
}

/// Two-sided Kolmogorov-Smirnov test of `data` against `cdf`.
///
/// Uses the Kolmogorov distribution for the p-value approximation.
pub fn ks_test(data: &[f64], cdf: impl Fn(f64) -> f64) -> KsResult {
    let n = data.len();
    if n == 0 {
        return KsResult {
            statistic: 0.0,
            p_value: 1.0,
        };
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let mut d = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f = cdf(x);
        let upper = (i + 1) as f64 / nf;
        let lower = i as f64 / nf;
        d = d.max((upper - f).abs()).max((f - lower).abs());
    }

    let p_value = kolmogorov_p(((nf).sqrt() + 0.12 + 0.11 / nf.sqrt()) * d);

    KsResult {
        statistic: d,
        p_value,
    }
}

/// KS test for discrete distributions (uses PMF-based CDF on integer grid).
pub fn ks_test_discrete(data: &[f64], cdf: impl Fn(u64) -> f64) -> KsResult {
    let n = data.len();
    if n == 0 {
        return KsResult {
            statistic: 0.0,
            p_value: 1.0,
        };
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let mut d = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let k = x.round() as u64;
        let f = cdf(k);
        let upper = (i + 1) as f64 / nf;
        let lower = i as f64 / nf;
        d = d.max((upper - f).abs()).max((f - lower).abs());
    }

    let p_value = kolmogorov_p(((nf).sqrt() + 0.12 + 0.11 / nf.sqrt()) * d);

    KsResult {
        statistic: d,
        p_value,
    }
}

/// Approximate p-value of the Kolmogorov distribution at `x`.
fn kolmogorov_p(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // P(K > x) = 2 Σ_{j=1}^∞ (−1)^{j+1} exp(−2j²x²)
    let mut sum = 0.0_f64;
    for j in 1_u32..=100 {
        let term = (-(2.0 * (j as f64).powi(2) * x * x)).exp();
        if j % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        if term < 1e-15 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

// ── Fit result ─────────────────────────────────────────────────────────────────

/// Summary of a distribution fit.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Distribution name (e.g. `"Normal"`, `"Gamma"`).
    pub name: String,
    /// Akaike Information Criterion (lower = better).
    pub aic: f64,
    /// Bayesian Information Criterion (lower = better).
    pub bic: f64,
    /// KS test statistic D.
    pub ks_statistic: f64,
    /// KS test p-value (higher = better fit).
    pub ks_p_value: f64,
}

// ── Continuous fitting ─────────────────────────────────────────────────────────

/// Fit all continuous distributions to `data` and return ranked results (by AIC).
///
/// Distributions that fail to fit (e.g. Beta when data are not in (0,1)) are silently skipped.
pub fn fit_all(data: &[f64]) -> StatsResult<Vec<FitResult>> {
    if data.is_empty() {
        return Err(StatsError::InvalidInput {
            message: "fit_all: data must not be empty".to_string(),
        });
    }

    let mut results: Vec<FitResult> = Vec::new();

    macro_rules! try_fit {
        ($dist_type:ty, $fit_expr:expr) => {
            if let Ok(dist) = $fit_expr {
                if let (Ok(aic), Ok(bic)) = (dist.aic(data), dist.bic(data)) {
                    if aic.is_finite() && bic.is_finite() {
                        let ks = ks_test(data, |x| dist.cdf(x).unwrap_or(0.0));
                        results.push(FitResult {
                            name: dist.name().to_string(),
                            aic,
                            bic,
                            ks_statistic: ks.statistic,
                            ks_p_value: ks.p_value,
                        });
                    }
                }
            }
        };
    }

    try_fit!(Normal, Normal::fit(data));
    try_fit!(
        Exponential,
        crate::distributions::exponential_distribution::Exponential::fit(data)
    );
    try_fit!(Uniform, Uniform::fit(data));
    try_fit!(Gamma, Gamma::fit(data));
    try_fit!(LogNormal, LogNormal::fit(data));
    try_fit!(Weibull, Weibull::fit(data));
    try_fit!(Beta, Beta::fit(data));
    try_fit!(StudentT, StudentT::fit(data));
    try_fit!(FDistribution, FDistribution::fit(data));
    try_fit!(ChiSquared, ChiSquared::fit(data));

    if results.is_empty() {
        return Err(StatsError::InvalidInput {
            message: "fit_all: no distribution could be fitted to the data".to_string(),
        });
    }

    results.sort_by(|a, b| {
        a.aic
            .partial_cmp(&b.aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

/// Fit all continuous distributions and return the best one (lowest AIC).
pub fn fit_best(data: &[f64]) -> StatsResult<FitResult> {
    let mut all = fit_all(data)?;
    Ok(all.remove(0))
}

// ── Discrete fitting ───────────────────────────────────────────────────────────

/// Fit all discrete distributions to integer `data` (passed as f64) and return ranked results.
///
/// Skips distributions that cannot be fitted.
pub fn fit_all_discrete(data: &[f64]) -> StatsResult<Vec<FitResult>> {
    if data.is_empty() {
        return Err(StatsError::InvalidInput {
            message: "fit_all_discrete: data must not be empty".to_string(),
        });
    }

    // Convert to u64 for discrete distributions
    let int_data: Vec<u64> = data.iter().map(|&x| x.round() as u64).collect();

    let mut results: Vec<FitResult> = Vec::new();

    macro_rules! try_fit_disc {
        ($fit_expr:expr) => {
            if let Ok(dist) = $fit_expr {
                if let (Ok(aic), Ok(bic)) = (dist.aic(&int_data), dist.bic(&int_data)) {
                    if aic.is_finite() && bic.is_finite() {
                        let ks = ks_test_discrete(data, |k| dist.cdf(k).unwrap_or(0.0));
                        results.push(FitResult {
                            name: dist.name().to_string(),
                            aic,
                            bic,
                            ks_statistic: ks.statistic,
                            ks_p_value: ks.p_value,
                        });
                    }
                }
            }
        };
    }

    try_fit_disc!(Poisson::fit(data));
    try_fit_disc!(Geometric::fit(data));
    try_fit_disc!(NegativeBinomial::fit(data));
    try_fit_disc!(Binomial::fit(data));

    if results.is_empty() {
        return Err(StatsError::InvalidInput {
            message: "fit_all_discrete: no distribution could be fitted to the data".to_string(),
        });
    }

    results.sort_by(|a, b| {
        a.aic
            .partial_cmp(&b.aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

/// Fit discrete distributions and return the best (lowest AIC).
pub fn fit_best_discrete(data: &[f64]) -> StatsResult<FitResult> {
    let mut all = fit_all_discrete(data)?;
    Ok(all.remove(0))
}

// ── Auto-detect and fit ────────────────────────────────────────────────────────

/// Automatically detect whether data is discrete or continuous, then fit all applicable
/// distributions and return the best match (lowest AIC).
///
/// # Examples
/// ```
/// use rs_stats::distributions::fitting::auto_fit;
///
/// let data = vec![1.2, 2.3, 1.8, 2.9, 1.5];
/// let best = auto_fit(&data).unwrap();
/// println!("Best fit: {}", best.name);
/// ```
pub fn auto_fit(data: &[f64]) -> StatsResult<FitResult> {
    match detect_data_type(data) {
        DataKind::Discrete => fit_best_discrete(data),
        DataKind::Continuous => fit_best(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_data_type_discrete() {
        assert_eq!(detect_data_type(&[0.0, 1.0, 2.0, 3.0]), DataKind::Discrete);
        assert_eq!(detect_data_type(&[0.0, 0.0, 1.0]), DataKind::Discrete);
    }

    #[test]
    fn test_detect_data_type_continuous() {
        assert_eq!(detect_data_type(&[0.5, 1.5, 2.3]), DataKind::Continuous);
        assert_eq!(detect_data_type(&[-1.0, 0.0, 1.0]), DataKind::Continuous);
        assert_eq!(detect_data_type(&[1.0, 2.5, 3.0]), DataKind::Continuous);
    }

    #[test]
    fn test_ks_test_uniform() {
        // Data from Uniform(0,1) should give large p-value against U(0,1) CDF
        let data: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let ks = ks_test(&data, |x| x.clamp(0.0, 1.0));
        assert!(ks.statistic < 0.15);
    }

    #[test]
    fn test_fit_all_returns_results() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1 + 0.5).collect();
        let results = fit_all(&data).unwrap();
        assert!(!results.is_empty());
        // Results sorted by AIC (ascending)
        for i in 1..results.len() {
            assert!(results[i].aic >= results[i - 1].aic);
        }
    }

    #[test]
    fn test_fit_best_normal_data() {
        // Data generated from N(5, 1)
        let data = vec![
            4.1, 5.2, 5.8, 4.7, 5.3, 4.9, 6.1, 4.5, 5.5, 5.0, 4.8, 5.1, 4.3, 5.7, 4.6, 5.4, 4.2,
            5.9, 5.2, 4.4,
        ];
        let best = fit_best(&data).unwrap();
        // Normal should win (or be competitive)
        assert!(best.aic.is_finite());
    }

    #[test]
    fn test_fit_all_discrete() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 2.0, 1.0, 0.0, 4.0];
        let results = fit_all_discrete(&data).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_auto_fit_continuous() {
        let data = vec![1.5, 2.3, 1.8, 2.1, 2.7, 1.9, 2.4, 2.0];
        let best = auto_fit(&data).unwrap();
        assert!(!best.name.is_empty());
    }

    #[test]
    fn test_auto_fit_discrete() {
        let data = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0];
        let best = auto_fit(&data).unwrap();
        assert!(!best.name.is_empty());
    }

    #[test]
    fn test_fit_all_empty_data() {
        assert!(fit_all(&[]).is_err());
    }
}
