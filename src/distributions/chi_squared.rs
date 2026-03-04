//! # Chi-Squared Distribution
//!
//! The Chi-squared distribution χ²(k) with k degrees of freedom is a special case
//! of the Gamma distribution: χ²(k) = Gamma(k/2, 1/2).
//!
//! **PDF**: f(x; k) = x^(k/2−1) · e^(−x/2) / (2^(k/2) · Γ(k/2)),  x ≥ 0
//!
//! **Mean**: k   **Variance**: 2k

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::{bisect_inverse_cdf, ln_gamma, regularized_incomplete_gamma};

/// Chi-squared distribution χ²(k).
///
/// # Examples
/// ```
/// use rs_stats::distributions::chi_squared::ChiSquared;
/// use rs_stats::distributions::traits::Distribution;
///
/// let c = ChiSquared::new(4.0).unwrap();
/// assert_eq!(c.mean(), 4.0);
/// assert_eq!(c.variance(), 8.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    /// Degrees of freedom k > 0
    pub k: f64,
}

impl ChiSquared {
    /// Creates a `χ²(k)` distribution.
    pub fn new(k: f64) -> StatsResult<Self> {
        if k <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "ChiSquared::new: k must be positive".to_string(),
            });
        }
        Ok(Self { k })
    }

    /// Method-of-moments MLE: k̂ = mean(data).
    ///
    /// Requires all data ≥ 0.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "ChiSquared::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x < 0.0) {
            return Err(StatsError::InvalidInput {
                message: "ChiSquared::fit: all data values must be non-negative".to_string(),
            });
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        Self::new(mean.max(0.01))
    }
}

impl Distribution for ChiSquared {
    fn name(&self) -> &str {
        "ChiSquared"
    }
    fn num_params(&self) -> usize {
        1
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        Ok(self.logpdf(x)?.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        let k = self.k;
        Ok((k / 2.0 - 1.0) * x.ln() - x / 2.0 - (k / 2.0) * 2_f64.ln() - ln_gamma(k / 2.0))
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        Ok(regularized_incomplete_gamma(self.k / 2.0, x / 2.0))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "ChiSquared::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let k = self.k;
        let hi = k + 10.0 * (2.0 * k).sqrt() + 50.0;
        Ok(bisect_inverse_cdf(
            |x| regularized_incomplete_gamma(k / 2.0, x / 2.0),
            p,
            0.0,
            hi,
        ))
    }

    fn mean(&self) -> f64 {
        self.k
    }

    fn variance(&self) -> f64 {
        2.0 * self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_squared_mean_variance() {
        let c = ChiSquared::new(6.0).unwrap();
        assert_eq!(c.mean(), 6.0);
        assert_eq!(c.variance(), 12.0);
    }

    #[test]
    fn test_chi_squared_cdf_zero() {
        let c = ChiSquared::new(4.0).unwrap();
        assert_eq!(c.cdf(0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_chi_squared_pdf_positive() {
        let c = ChiSquared::new(2.0).unwrap();
        // χ²(2) = Exp(1/2): pdf(x) = 0.5 * e^{-x/2}
        let expected = 0.5 * (-1.0_f64).exp();
        assert!((c.pdf(2.0).unwrap() - expected).abs() < 1e-8);
    }

    #[test]
    fn test_chi_squared_inverse_cdf_roundtrip() {
        let c = ChiSquared::new(5.0).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = c.inverse_cdf(p).unwrap();
            let p_back = c.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn test_chi_squared_fit() {
        let data = vec![3.0, 5.0, 4.0, 6.0, 2.0, 4.5, 5.5, 3.5];
        let c = ChiSquared::fit(&data).unwrap();
        let expected_k = data.iter().sum::<f64>() / data.len() as f64;
        assert!((c.k - expected_k).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_invalid() {
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
    }
}
