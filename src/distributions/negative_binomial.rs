//! # Negative Binomial Distribution
//!
//! The Negative Binomial distribution NegBinom(r, p) models the number of failures
//! before achieving r successes in Bernoulli trials with success probability p.
//! It also serves as the canonical model for **overdispersed count data** (variance > mean).
//!
//! Support: k = 0, 1, 2, …  (number of failures)
//!
//! **PMF**: P(X = k) = C(k+r−1, k) · p^r · (1−p)^k
//!
//! **Mean**: r(1−p)/p   **Variance**: r(1−p)/p²
//!
//! ## When to use over Poisson
//!
//! When count data has **variance > mean** (overdispersion), Poisson underfits.
//! Use Negative Binomial — it adds a free parameter r to absorb the extra variability.
//! Common in healthcare where patients are heterogeneous (different baseline risks).
//!
//! ## Medical applications
//!
//! - **Hospital readmissions** before stable remission (heterogeneous patient risk)
//! - **Overdispersed adverse event counts** in pharmacovigilance
//! - **Number of disease recurrences** before sustained response
//! - **Emergency department visits** per patient per year (high inter-patient variability)
//!
//! ## Example — re-admissions before remission
//!
//! ```rust
//! use rs_stats::distributions::negative_binomial::NegativeBinomial;
//! use rs_stats::DiscreteDistribution;
//!
//! // Re-admissions data across 20 patients — variance >> mean → overdispersed
//! let admissions = vec![
//!     0.0, 2.0, 1.0, 5.0, 3.0, 0.0, 4.0, 1.0, 2.0, 6.0,
//!     1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 0.0, 2.0, 3.0, 1.0,
//! ];
//! let nb = NegativeBinomial::fit(&admissions).unwrap();
//! println!("NegBin(r={:.2}, p={:.3})", nb.r, nb.p);
//! println!("P(0 re-admissions) = {:.1}%", nb.pmf(0).unwrap() * 100.0);
//! ```

use crate::distributions::traits::DiscreteDistribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::ln_gamma;
use serde::{Deserialize, Serialize};

/// Negative Binomial distribution NegBinom(r, p).
///
/// # Examples
/// ```
/// use rs_stats::distributions::negative_binomial::NegativeBinomial;
/// use rs_stats::distributions::traits::DiscreteDistribution;
///
/// let nb = NegativeBinomial::new(5.0, 0.5).unwrap();
/// assert!((nb.mean() - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NegativeBinomial {
    /// Number of successes r > 0 (can be non-integer, i.e. the overdispersion parameter)
    pub r: f64,
    /// Success probability p ∈ (0, 1)
    pub p: f64,
}

impl NegativeBinomial {
    /// Creates a `NegBinom(r, p)` distribution.
    pub fn new(r: f64, p: f64) -> StatsResult<Self> {
        if r <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "NegativeBinomial::new: r must be positive".to_string(),
            });
        }
        if !(0.0 < p && p < 1.0) {
            return Err(StatsError::InvalidInput {
                message: "NegativeBinomial::new: p must be in (0, 1)".to_string(),
            });
        }
        Ok(Self { r, p })
    }

    /// Method-of-moments fitting.
    ///
    /// - mean = r(1−p)/p  → p = mean/variance (requires variance > mean)
    /// - r = mean² / (variance − mean)
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "NegativeBinomial::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x < 0.0 || x.fract() != 0.0) {
            return Err(StatsError::InvalidInput {
                message: "NegativeBinomial::fit: all data values must be non-negative integers"
                    .to_string(),
            });
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        if variance <= mean {
            // Data appears Poisson-like; fall back to large r (approximates Poisson)
            return Self::new(mean.max(0.01) * 10.0, 1.0 - 1.0 / 11.0);
        }

        let p = mean / variance;
        let r = mean * p / (1.0 - p);
        Self::new(r.max(0.01), p.clamp(1e-9, 1.0 - 1e-9))
    }
}

impl DiscreteDistribution for NegativeBinomial {
    fn name(&self) -> &str {
        "NegativeBinomial"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pmf(&self, k: u64) -> StatsResult<f64> {
        Ok(self.logpmf(k)?.exp())
    }

    fn logpmf(&self, k: u64) -> StatsResult<f64> {
        let kf = k as f64;
        // log C(k+r-1, k) = ln_gamma(k+r) - ln_gamma(r) - ln_gamma(k+1)
        let log_binom = ln_gamma(kf + self.r) - ln_gamma(self.r) - ln_gamma(kf + 1.0);
        Ok(log_binom + self.r * self.p.ln() + kf * (1.0 - self.p).ln())
    }

    fn cdf(&self, k: u64) -> StatsResult<f64> {
        // Sum PMF from 0 to k
        let mut sum = 0.0_f64;
        for i in 0..=k {
            sum += self.pmf(i)?;
            // Early exit if essentially 1
            if sum >= 1.0 - 1e-15 {
                return Ok(1.0);
            }
        }
        Ok(sum.clamp(0.0, 1.0))
    }

    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }

    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neg_binom_mean_variance() {
        let nb = NegativeBinomial::new(5.0, 0.5).unwrap();
        assert!((nb.mean() - 5.0).abs() < 1e-10);
        assert!((nb.variance() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_neg_binom_pmf_k0() {
        // P(X=0) = p^r
        let nb = NegativeBinomial::new(2.0, 0.5).unwrap();
        assert!((nb.pmf(0).unwrap() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_neg_binom_cdf_monotone() {
        let nb = NegativeBinomial::new(3.0, 0.4).unwrap();
        let mut prev = 0.0;
        for k in 0..20 {
            let c = nb.cdf(k).unwrap();
            assert!(c >= prev, "CDF not monotone at k={k}");
            prev = c;
        }
    }

    #[test]
    fn test_neg_binom_fit() {
        let data = vec![0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 4.0, 1.0, 2.0];
        let nb = NegativeBinomial::fit(&data).unwrap();
        assert!(nb.r > 0.0 && nb.p > 0.0 && nb.p < 1.0);
    }

    #[test]
    fn test_neg_binom_invalid() {
        assert!(NegativeBinomial::new(0.0, 0.5).is_err());
        assert!(NegativeBinomial::new(1.0, 0.0).is_err());
        assert!(NegativeBinomial::new(1.0, 1.0).is_err());
    }
}
