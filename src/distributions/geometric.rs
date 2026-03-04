//! # Geometric Distribution
//!
//! The Geometric distribution models the number of Bernoulli trials until (and
//! including) the first success, where each trial has constant success probability p.
//!
//! Support: k = 1, 2, 3, …
//!
//! **PMF**: P(X = k) = (1−p)^(k−1) · p
//!
//! **Mean**: 1/p   **Variance**: (1−p)/p²
//!
//! ## Medical applications
//!
//! - **Screening programs**: number of colonoscopy sessions until a polyp is detected
//! - **Treatment attempts**: number of chemotherapy cycles until remission is achieved
//! - **Procedure success**: number of lumbar puncture attempts until CSF is obtained
//! - **Vaccination campaigns**: number of contact attempts until a patient is reached
//!
//! ## Example — colonoscopy screening
//!
//! ```rust
//! use rs_stats::distributions::geometric::Geometric;
//! use rs_stats::DiscreteDistribution;
//!
//! // P(detecting a polyp per colonoscopy session) = 0.18
//! let screening = Geometric::new(0.18).unwrap();
//! println!("E[sessions to detect] = {:.1}", screening.mean());   // ≈ 5.6
//! println!("P(detected ≤ 3)       = {:.1}%", screening.cdf(3).unwrap() * 100.0);
//! ```

use crate::distributions::traits::DiscreteDistribution;
use crate::error::{StatsError, StatsResult};

/// Geometric distribution Geometric(p).
///
/// # Examples
/// ```
/// use rs_stats::distributions::geometric::Geometric;
/// use rs_stats::distributions::traits::DiscreteDistribution;
///
/// let g = Geometric::new(0.25).unwrap();
/// assert!((g.mean() - 4.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    /// Success probability p ∈ (0, 1]
    pub p: f64,
}

impl Geometric {
    /// Creates a `Geometric(p)` distribution.
    pub fn new(p: f64) -> StatsResult<Self> {
        if !(0.0 < p && p <= 1.0) {
            return Err(StatsError::InvalidInput {
                message: "Geometric::new: p must be in (0, 1]".to_string(),
            });
        }
        Ok(Self { p })
    }

    /// MLE: p̂ = 1 / mean(data).
    ///
    /// Data must be positive integers ≥ 1 (passed as f64 values that are whole numbers).
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Geometric::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x < 1.0 || x.fract() != 0.0) {
            return Err(StatsError::InvalidInput {
                message: "Geometric::fit: all data values must be positive integers (≥ 1)"
                    .to_string(),
            });
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        Self::new((1.0 / mean).clamp(1e-15, 1.0))
    }
}

impl DiscreteDistribution for Geometric {
    fn name(&self) -> &str {
        "Geometric"
    }
    fn num_params(&self) -> usize {
        1
    }

    fn pmf(&self, k: u64) -> StatsResult<f64> {
        if k == 0 {
            return Ok(0.0);
        }
        Ok(self.p * (1.0 - self.p).powi((k - 1) as i32))
    }

    fn logpmf(&self, k: u64) -> StatsResult<f64> {
        if k == 0 {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(self.p.ln() + (k - 1) as f64 * (1.0 - self.p).ln())
    }

    fn cdf(&self, k: u64) -> StatsResult<f64> {
        if k == 0 {
            return Ok(0.0);
        }
        // CDF(k) = 1 - (1-p)^k
        Ok(1.0 - (1.0 - self.p).powi(k as i32))
    }

    fn mean(&self) -> f64 {
        1.0 / self.p
    }

    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_mean_variance() {
        let g = Geometric::new(0.25).unwrap();
        assert!((g.mean() - 4.0).abs() < 1e-10);
        assert!((g.variance() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_pmf_k1() {
        // P(X=1) = p
        let g = Geometric::new(0.5).unwrap();
        assert!((g.pmf(1).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_cdf_large_k() {
        let g = Geometric::new(0.5).unwrap();
        // CDF should approach 1
        assert!(g.cdf(100).unwrap() > 0.999_999);
    }

    #[test]
    fn test_geometric_logpmf() {
        let g = Geometric::new(0.3).unwrap();
        let pmf = g.pmf(3).unwrap();
        let logpmf = g.logpmf(3).unwrap();
        assert!((logpmf - pmf.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_fit() {
        let data = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 4.0, 1.0];
        let g = Geometric::fit(&data).unwrap();
        let expected_p = data.len() as f64 / data.iter().sum::<f64>();
        assert!((g.p - expected_p).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_invalid() {
        assert!(Geometric::new(0.0).is_err());
        assert!(Geometric::new(-0.1).is_err());
        assert!(Geometric::new(1.1).is_err());
    }
}
