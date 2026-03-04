//! # Distribution Traits
//!
//! Defines the [`Distribution`] and [`DiscreteDistribution`] traits that provide
//! a unified interface for all statistical distributions in this crate.
//!
//! ## Usage
//!
//! ```
//! use rs_stats::distributions::traits::Distribution;
//! use rs_stats::distributions::normal_distribution::Normal;
//!
//! let n = Normal::new(0.0, 1.0).unwrap();
//! let pdf = n.pdf(0.0).unwrap();
//! assert!((pdf - 0.398_942_280_4).abs() < 1e-8);
//! ```

use crate::error::StatsResult;

// ── Continuous distributions ───────────────────────────────────────────────────

/// Unified interface for continuous probability distributions.
///
/// All methods return `StatsResult` to propagate domain errors (e.g. `p ∉ [0,1]`).
///
/// The trait is **object-safe**: `Box<dyn Distribution>` works at runtime.
/// The `fit` associated function is intentionally *not* part of the trait to preserve
/// object safety; each concrete type exposes `Dist::fit(data)` directly.
pub trait Distribution {
    /// Human-readable distribution name, e.g. `"Normal"`.
    fn name(&self) -> &str;

    /// Number of free parameters (used when computing AIC / BIC).
    fn num_params(&self) -> usize;

    /// Probability density function f(x).
    fn pdf(&self, x: f64) -> StatsResult<f64>;

    /// Natural logarithm of the PDF: ln f(x).
    ///
    /// Default implementation delegates to `pdf`; override for numerical stability.
    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        self.pdf(x).map(|p| p.ln())
    }

    /// Cumulative distribution function F(x) = P(X ≤ x).
    fn cdf(&self, x: f64) -> StatsResult<f64>;

    /// Quantile (inverse CDF): find x such that F(x) = p.
    fn inverse_cdf(&self, p: f64) -> StatsResult<f64>;

    /// Mean (expected value) μ.
    fn mean(&self) -> f64;

    /// Variance σ².
    fn variance(&self) -> f64;

    /// Standard deviation σ = √(variance).
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sum of log-likelihoods: Σ ln f(xᵢ).
    fn log_likelihood(&self, data: &[f64]) -> StatsResult<f64> {
        let mut ll = 0.0_f64;
        for &x in data {
            ll += self.logpdf(x)?;
        }
        Ok(ll)
    }

    /// Akaike Information Criterion: AIC = 2k − 2·ln(L̂).
    fn aic(&self, data: &[f64]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        Ok(2.0 * self.num_params() as f64 - 2.0 * ll)
    }

    /// Bayesian Information Criterion: BIC = k·ln(n) − 2·ln(L̂).
    fn bic(&self, data: &[f64]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        let n = data.len() as f64;
        Ok(self.num_params() as f64 * n.ln() - 2.0 * ll)
    }
}

// ── Discrete distributions ─────────────────────────────────────────────────────

/// Unified interface for discrete probability distributions.
///
/// Works with non-negative integer observations represented as `u64`.
///
/// Object-safe: `Box<dyn DiscreteDistribution>` is valid.
pub trait DiscreteDistribution {
    /// Human-readable distribution name.
    fn name(&self) -> &str;

    /// Number of free parameters (used for AIC / BIC).
    fn num_params(&self) -> usize;

    /// Probability mass function P(X = k).
    fn pmf(&self, k: u64) -> StatsResult<f64>;

    /// Natural logarithm of the PMF: ln P(X = k).
    ///
    /// Default: delegates to `pmf`; override for stability when p is tiny.
    fn logpmf(&self, k: u64) -> StatsResult<f64> {
        self.pmf(k).map(|p| p.ln())
    }

    /// Cumulative distribution function P(X ≤ k).
    fn cdf(&self, k: u64) -> StatsResult<f64>;

    /// Mean (expected value) μ.
    fn mean(&self) -> f64;

    /// Variance σ².
    fn variance(&self) -> f64;

    /// Standard deviation σ = √(variance).
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sum of log-PMFs: Σ ln P(X = kᵢ).
    fn log_likelihood(&self, data: &[u64]) -> StatsResult<f64> {
        let mut ll = 0.0_f64;
        for &k in data {
            ll += self.logpmf(k)?;
        }
        Ok(ll)
    }

    /// AIC = 2k − 2·ln(L̂).
    fn aic(&self, data: &[u64]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        Ok(2.0 * self.num_params() as f64 - 2.0 * ll)
    }

    /// BIC = k·ln(n) − 2·ln(L̂).
    fn bic(&self, data: &[u64]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        let n = data.len() as f64;
        Ok(self.num_params() as f64 * n.ln() - 2.0 * ll)
    }
}
