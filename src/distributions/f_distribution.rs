//! # F-Distribution
//!
//! The F-distribution F(d1, d2) with d1 numerator and d2 denominator degrees
//! of freedom models ratios of chi-squared random variables.
//!
//! **PDF**: f(x; d1, d2) = √((d1·x)^d1 · d2^d2 / (d1·x + d2)^(d1+d2)) / (x · B(d1/2, d2/2))
//!
//! **Mean**: d2 / (d2 − 2)  for d2 > 2

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::{bisect_inverse_cdf, ln_beta, regularized_incomplete_beta};

/// F-distribution F(d1, d2).
///
/// # Examples
/// ```
/// use rs_stats::distributions::f_distribution::FDistribution;
/// use rs_stats::distributions::traits::Distribution;
///
/// let f = FDistribution::new(5.0, 10.0).unwrap();
/// assert!((f.mean() - 10.0 / 8.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FDistribution {
    /// Numerator degrees of freedom d1 > 0
    pub d1: f64,
    /// Denominator degrees of freedom d2 > 0
    pub d2: f64,
}

impl FDistribution {
    /// Creates an `F(d1, d2)` distribution.
    pub fn new(d1: f64, d2: f64) -> StatsResult<Self> {
        if d1 <= 0.0 || d2 <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "FDistribution::new: d1 and d2 must be positive".to_string(),
            });
        }
        Ok(Self { d1, d2 })
    }

    /// Method-of-moments fitting from data > 0.
    ///
    /// Estimates d2 from the mean (requires mean > 1) and d1 from the variance.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "FDistribution::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x <= 0.0) {
            return Err(StatsError::InvalidInput {
                message: "FDistribution::fit: all data values must be positive".to_string(),
            });
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        // E[X] = d2/(d2-2) → d2 = 2*mean/(mean-1)
        if mean <= 1.0 {
            // Can't estimate; fall back to typical defaults
            return Self::new(2.0, 10.0);
        }
        let d2 = (2.0 * mean / (mean - 1.0)).max(2.01);

        // Var[X] = 2d2²(d1+d2-2) / (d1(d2-2)²(d2-4))
        // Solve for d1:
        //   d1 = 2*d2²*(d2-2) / (variance*(d2-2)²*(d2-4)/d2² + 2*(d2-2)*(d2-4)*d2 / d2²)
        // Simplified from the quadratic:
        let d2m2 = d2 - 2.0;
        let d2m4 = (d2 - 4.0).max(0.01);
        // Var ≈ 2d2²(d1+d2-2) / (d1 * d2m2² * d2m4)
        // d1 * variance * d2m2² * d2m4 = 2d2²(d1 + d2 - 2)
        // d1 * (variance * d2m2² * d2m4 - 2d2²) = 2d2²(d2 - 2)
        let numerator = 2.0 * d2 * d2 * d2m2;
        let denominator = variance * d2m2 * d2m2 * d2m4 - 2.0 * d2 * d2;
        let d1 = if denominator > 0.0 {
            (numerator / denominator).max(0.01)
        } else {
            2.0
        };

        Self::new(d1, d2)
    }
}

impl Distribution for FDistribution {
    fn name(&self) -> &str {
        "F"
    }
    fn num_params(&self) -> usize {
        2
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
        let d1 = self.d1;
        let d2 = self.d2;
        let log_pdf = 0.5 * (d1 * (d1 * x).ln() + d2 * d2.ln() - (d1 + d2) * (d1 * x + d2).ln())
            - x.ln()
            - ln_beta(d1 / 2.0, d2 / 2.0);
        Ok(log_pdf)
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        // I_{d1*x/(d1*x + d2)}(d1/2, d2/2)
        let t = self.d1 * x / (self.d1 * x + self.d2);
        Ok(regularized_incomplete_beta(self.d1 / 2.0, self.d2 / 2.0, t))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "FDistribution::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let d1 = self.d1;
        let d2 = self.d2;
        let mean = if d2 > 2.0 { d2 / (d2 - 2.0) } else { 5.0 };
        let hi = mean * 20.0;
        Ok(bisect_inverse_cdf(
            |x| {
                if x <= 0.0 {
                    return 0.0;
                }
                let t = d1 * x / (d1 * x + d2);
                regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, t)
            },
            p,
            0.0,
            hi,
        ))
    }

    fn mean(&self) -> f64 {
        if self.d2 > 2.0 {
            self.d2 / (self.d2 - 2.0)
        } else {
            f64::INFINITY
        }
    }

    fn variance(&self) -> f64 {
        let d1 = self.d1;
        let d2 = self.d2;
        if d2 > 4.0 {
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0))
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_mean() {
        let f = FDistribution::new(5.0, 10.0).unwrap();
        assert!((f.mean() - 10.0 / 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_f_cdf_zero() {
        let f = FDistribution::new(3.0, 6.0).unwrap();
        assert_eq!(f.cdf(0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_f_pdf_positive() {
        let f = FDistribution::new(2.0, 2.0).unwrap();
        let p = f.pdf(1.0).unwrap();
        assert!(p > 0.0 && p.is_finite());
    }

    #[test]
    fn test_f_inverse_cdf_roundtrip() {
        let f = FDistribution::new(5.0, 10.0).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = f.inverse_cdf(p).unwrap();
            let p_back = f.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-5, "p={p}");
        }
    }

    #[test]
    fn test_f_invalid() {
        assert!(FDistribution::new(0.0, 5.0).is_err());
        assert!(FDistribution::new(5.0, 0.0).is_err());
    }
}
