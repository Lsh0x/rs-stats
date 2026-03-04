//! # Beta Distribution
//!
//! The Beta distribution Beta(α, β) is a continuous distribution on [0, 1],
//! commonly used to model probabilities and proportions.
//!
//! **PDF**: f(x; α, β) = x^(α−1) · (1−x)^(β−1) / B(α, β),  x ∈ [0, 1]
//!
//! **MLE**: method-of-moments — μ̂ and σ̂² from data → α̂, β̂.

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::{bisect_inverse_cdf, ln_beta, regularized_incomplete_beta};

/// Beta distribution Beta(α, β).
///
/// # Examples
/// ```
/// use rs_stats::distributions::beta::Beta;
/// use rs_stats::distributions::traits::Distribution;
///
/// let b = Beta::new(2.0, 5.0).unwrap();
/// assert!((b.mean() - 2.0 / 7.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    /// Shape parameter α > 0
    pub alpha: f64,
    /// Shape parameter β > 0
    pub beta: f64,
}

impl Beta {
    /// Creates a `Beta(α, β)` distribution. Both parameters must be positive.
    pub fn new(alpha: f64, beta: f64) -> StatsResult<Self> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Beta::new: alpha and beta must be positive".to_string(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// MLE via method of moments from data in [0, 1].
    ///
    /// Requires all data in (0, 1). Estimates α and β from sample mean and variance.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Beta::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x <= 0.0 || x >= 1.0) {
            return Err(StatsError::InvalidInput {
                message: "Beta::fit: all data values must be in (0, 1)".to_string(),
            });
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        // Method of moments: α = mean*(mean*(1-mean)/var - 1), β = (1-mean)*...
        let common = mean * (1.0 - mean) / variance - 1.0;
        let alpha = mean * common;
        let beta = (1.0 - mean) * common;
        Self::new(alpha, beta)
    }
}

impl Distribution for Beta {
    fn name(&self) -> &str {
        "Beta"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&x) {
            return Ok(0.0);
        }
        if x == 0.0 {
            return Ok(if self.alpha >= 1.0 {
                0.0
            } else {
                f64::INFINITY
            });
        }
        if x == 1.0 {
            return Ok(if self.beta >= 1.0 { 0.0 } else { f64::INFINITY });
        }
        let log_pdf = (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln()
            - ln_beta(self.alpha, self.beta);
        Ok(log_pdf.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 || x >= 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(
            (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln()
                - ln_beta(self.alpha, self.beta),
        )
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        if x >= 1.0 {
            return Ok(1.0);
        }
        Ok(regularized_incomplete_beta(self.alpha, self.beta, x))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Beta::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(1.0);
        }
        let alpha = self.alpha;
        let beta = self.beta;
        Ok(bisect_inverse_cdf(
            |x| regularized_incomplete_beta(alpha, beta, x),
            p,
            0.0,
            1.0,
        ))
    }

    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        self.alpha * self.beta / (s * s * (s + 1.0))
    }
}

// ── Log-likelihood with analytically stable logpdf ────────────────────────────
// (default impl already calls logpdf, which is overridden above)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_mean_variance() {
        let b = Beta::new(2.0, 5.0).unwrap();
        assert!((b.mean() - 2.0 / 7.0).abs() < 1e-10);
        let expected_var = 2.0 * 5.0 / (49.0 * 8.0);
        assert!((b.variance() - expected_var).abs() < 1e-10);
    }

    #[test]
    fn test_beta_pdf_at_mean() {
        let b = Beta::new(1.0, 1.0).unwrap(); // Uniform on [0,1]
        assert!((b.pdf(0.5).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_cdf_bounds() {
        let b = Beta::new(2.0, 3.0).unwrap();
        assert_eq!(b.cdf(0.0).unwrap(), 0.0);
        assert_eq!(b.cdf(1.0).unwrap(), 1.0);
    }

    #[test]
    fn test_beta_inverse_cdf_roundtrip() {
        let b = Beta::new(2.0, 3.0).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = b.inverse_cdf(p).unwrap();
            let p_back = b.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-7, "p={p}: roundtrip failed");
        }
    }

    #[test]
    fn test_beta_fit() {
        // Fit from data generated from Beta(2, 5)
        let data = vec![0.1, 0.2, 0.15, 0.25, 0.3, 0.18, 0.22, 0.12, 0.28, 0.16];
        let b = Beta::fit(&data).unwrap();
        // Mean of the data ≈ mean of Beta(α, β)
        let data_mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!((b.mean() - data_mean).abs() < 1e-10);
    }

    #[test]
    fn test_beta_invalid_params() {
        assert!(Beta::new(0.0, 1.0).is_err());
        assert!(Beta::new(1.0, 0.0).is_err());
        assert!(Beta::new(-1.0, 1.0).is_err());
    }
}
