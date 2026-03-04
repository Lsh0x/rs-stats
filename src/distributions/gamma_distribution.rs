//! # Gamma Distribution
//!
//! The Gamma distribution Gamma(α, β) (shape α, rate β) is a continuous distribution
//! over (0, ∞), generalising the exponential and chi-squared distributions.
//!
//! **PDF**: f(x; α, β) = β^α · x^(α−1) · e^(−βx) / Γ(α),  x > 0
//!
//! **Mean**: α/β   **Variance**: α/β²

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::{bisect_inverse_cdf, ln_gamma, regularized_incomplete_gamma};

/// Gamma distribution Gamma(α, β) with shape α and rate β.
///
/// # Examples
/// ```
/// use rs_stats::distributions::gamma_distribution::Gamma;
/// use rs_stats::distributions::traits::Distribution;
///
/// let g = Gamma::new(2.0, 1.0).unwrap();
/// assert!((g.mean() - 2.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    /// Shape parameter α > 0
    pub alpha: f64,
    /// Rate parameter β > 0  (= 1/scale)
    pub beta: f64,
}

impl Gamma {
    /// Creates a `Gamma(α, β)` distribution.
    pub fn new(alpha: f64, beta: f64) -> StatsResult<Self> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Gamma::new: alpha and beta must be positive".to_string(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// MLE fitting via Choi-Wette approximation for α, then β = α / mean.
    ///
    /// Requires all data > 0.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Gamma::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x <= 0.0) {
            return Err(StatsError::InvalidInput {
                message: "Gamma::fit: all data values must be positive".to_string(),
            });
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let log_mean = data.iter().map(|&x| x.ln()).sum::<f64>() / n;
        // s = ln(mean) - mean(ln(x))  (always ≥ 0)
        let s = mean.ln() - log_mean;
        // Choi-Wette approximation for MLE of α
        let alpha = if s > 0.0 {
            (3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s)
        } else {
            1.0
        };
        let beta = alpha / mean;
        Self::new(alpha, beta)
    }
}

impl Distribution for Gamma {
    fn name(&self) -> &str {
        "Gamma"
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
        Ok(self.alpha * self.beta.ln() + (self.alpha - 1.0) * x.ln()
            - self.beta * x
            - ln_gamma(self.alpha))
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        Ok(regularized_incomplete_gamma(self.alpha, self.beta * x))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Gamma::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let alpha = self.alpha;
        let beta = self.beta;
        // Upper bound: mean + 10*std_dev should cover virtually all mass
        let hi = (alpha / beta) + 10.0 * (alpha / beta / beta).sqrt();
        Ok(bisect_inverse_cdf(
            |x| regularized_incomplete_gamma(alpha, beta * x),
            p,
            0.0,
            hi.max(1.0),
        ))
    }

    fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_mean_variance() {
        let g = Gamma::new(3.0, 2.0).unwrap();
        assert!((g.mean() - 1.5).abs() < 1e-10);
        assert!((g.variance() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_pdf_positive() {
        let g = Gamma::new(2.0, 1.0).unwrap();
        let p = g.pdf(1.0).unwrap();
        // Gamma(2,1): pdf(1) = e^{-1} ≈ 0.36788
        assert!((p - (-1.0_f64).exp()).abs() < 1e-8);
    }

    #[test]
    fn test_gamma_cdf_zero() {
        let g = Gamma::new(2.0, 1.0).unwrap();
        assert_eq!(g.cdf(0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_gamma_inverse_cdf_roundtrip() {
        let g = Gamma::new(3.0, 0.5).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = g.inverse_cdf(p).unwrap();
            let p_back = g.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-6, "p={p}: roundtrip failed");
        }
    }

    #[test]
    fn test_gamma_fit() {
        // Gamma(2, 1): mean=2, data ≈ exponentially distributed
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.8, 2.2, 0.8, 3.2, 1.0];
        let g = Gamma::fit(&data).unwrap();
        let data_mean = data.iter().sum::<f64>() / data.len() as f64;
        // Fitted mean must equal data mean
        assert!((g.mean() - data_mean).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_invalid() {
        assert!(Gamma::new(0.0, 1.0).is_err());
        assert!(Gamma::new(1.0, -1.0).is_err());
    }
}
