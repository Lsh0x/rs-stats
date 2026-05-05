//! # Distribution trait
//!
//! Single unified [`Distribution`] trait covering both continuous and
//! discrete probability distributions, parameterised by an associated
//! support type [`Distribution::X`] — `f64` for continuous, `u64` for
//! discrete. The previous `Distribution` / `DiscreteDistribution` split
//! is gone in v3.0.
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
//!
//! For polymorphism, parameterise on the support type:
//!
//! ```
//! use rs_stats::Distribution;
//! use rs_stats::distributions::normal_distribution::Normal;
//! use rs_stats::distributions::lognormal::LogNormal;
//!
//! fn pick(skewed: bool) -> Box<dyn Distribution<X = f64>> {
//!     if skewed {
//!         Box::new(LogNormal::new(1.0, 0.5).unwrap())
//!     } else {
//!         Box::new(Normal::new(80.0, 10.0).unwrap())
//!     }
//! }
//! ```

use crate::error::{StatsError, StatsResult};

/// Unified probability-distribution interface.
///
/// All methods return [`StatsResult`] to propagate domain errors (e.g.
/// `p ∉ [0, 1]`). The trait is **object-safe** with the support type
/// fixed: `Box<dyn Distribution<X = f64>>` and
/// `Box<dyn Distribution<X = u64>>` both work.
///
/// `fit` is intentionally not part of the trait (it would break object
/// safety); each concrete type exposes `Dist::fit(data)` directly.
pub trait Distribution {
    /// The support type: `f64` for continuous distributions, `u64` for
    /// discrete distributions.
    type X: Copy;

    /// Human-readable distribution name, e.g. `"Normal"`.
    fn name(&self) -> &str;

    /// Number of free parameters — used for AIC / BIC.
    fn num_params(&self) -> usize;

    /// Probability density (continuous) or probability mass (discrete) at `x`.
    fn pdf(&self, x: Self::X) -> StatsResult<f64>;

    /// `ln pdf(x)`. Default: `pdf(x).ln()`. Override for stability when
    /// `pdf` underflows (e.g. extreme tails of LogNormal).
    fn logpdf(&self, x: Self::X) -> StatsResult<f64> {
        self.pdf(x).map(|p| p.ln())
    }

    /// Probability mass — alias for [`pdf`](Self::pdf), kept for natural
    /// reading on discrete distributions and v2.x source-compatibility.
    fn pmf(&self, x: Self::X) -> StatsResult<f64> {
        self.pdf(x)
    }

    /// Log-PMF — alias for [`logpdf`](Self::logpdf).
    fn logpmf(&self, x: Self::X) -> StatsResult<f64> {
        self.logpdf(x)
    }

    /// Cumulative distribution function `F(x) = P(X ≤ x)`.
    fn cdf(&self, x: Self::X) -> StatsResult<f64>;

    /// Quantile / inverse CDF: smallest `x` such that `F(x) ≥ p`.
    fn inverse_cdf(&self, p: f64) -> StatsResult<Self::X>;

    /// Mean (expected value) `μ`.
    fn mean(&self) -> f64;

    /// Variance `σ²`.
    fn variance(&self) -> f64;

    /// Standard deviation `σ = √(variance)`.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sum of log-likelihoods `Σ ln pdf(xᵢ)`. Falls back to
    /// [`StatsError`] if any point is out of support.
    fn log_likelihood(&self, data: &[Self::X]) -> StatsResult<f64> {
        let mut ll = 0.0_f64;
        for &x in data {
            ll += self.logpdf(x)?;
        }
        Ok(ll)
    }

    /// Infallible bulk log-likelihood. Out-of-support points contribute
    /// `f64::NEG_INFINITY` (as in scipy). Implementations should override
    /// this when they can express the inner loop without per-point
    /// branching, so LLVM can autovectorise it.
    fn log_likelihood_fast(&self, data: &[Self::X]) -> f64 {
        let mut ll = 0.0_f64;
        for &x in data {
            ll += self.logpdf(x).unwrap_or(f64::NEG_INFINITY);
        }
        ll
    }

    /// Akaike Information Criterion: `2k − 2·ln(L̂)`.
    fn aic(&self, data: &[Self::X]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        Ok(2.0 * self.num_params() as f64 - 2.0 * ll)
    }

    /// Bayesian Information Criterion: `k·ln(n) − 2·ln(L̂)`.
    fn bic(&self, data: &[Self::X]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        let n = data.len() as f64;
        Ok(self.num_params() as f64 * n.ln() - 2.0 * ll)
    }
}

// ── Helpers shared across discrete distributions ──────────────────────────────

/// Generic discrete inverse-CDF: smallest `k ≥ 0` such that `cdf(k) ≥ p`.
///
/// Works on any monotone non-decreasing CDF over `u64`. Phase 1
/// exponential-doubling brackets the answer, Phase 2 binary-searches.
/// Discrete distributions whose `inverse_cdf` has no closed form delegate
/// to this helper.
pub(crate) fn discrete_inverse_cdf_search(
    p: f64,
    cdf: impl Fn(u64) -> StatsResult<f64>,
) -> StatsResult<u64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::InvalidInput {
            message: format!("inverse_cdf: p must be in [0, 1], got {p}"),
        });
    }
    if p == 0.0 {
        return Ok(0);
    }
    let mut hi: u64 = 1;
    while cdf(hi)? < p {
        hi = hi.saturating_mul(2);
        if hi == u64::MAX {
            return Err(StatsError::NumericalError {
                message: "inverse_cdf: quantile exceeds u64::MAX".to_string(),
            });
        }
    }
    let mut lo: u64 = 0;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf(mid)? < p {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// Backwards-compatibility alias. `DiscreteDistribution` was a separate
/// trait in v2.x; in v3.0 every distribution implements the unified
/// [`Distribution`] trait. This alias matches discrete distributions
/// (those whose support is `u64`) and keeps existing imports working.
pub trait DiscreteDistribution: Distribution<X = u64> {}
impl<T: Distribution<X = u64>> DiscreteDistribution for T {}
