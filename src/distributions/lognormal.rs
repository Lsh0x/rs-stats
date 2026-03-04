//! # Log-Normal Distribution
//!
//! If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ²).
//! The distribution is defined for x > 0 and is right-skewed.
//!
//! **PDF**: f(x) = 1 / (x σ √(2π)) · exp(−(ln x − μ)² / (2σ²))
//!
//! **MLE**: μ̂ = mean(ln data),  σ̂ = pop-std(ln data)  (exact MLE)
//!
//! **Median** = exp(μ)  (more informative than mean for skewed data)
//!
//! ## When to use
//!
//! Log-Normal arises naturally when a positive quantity is the *product* of many
//! independent factors, or whenever the natural logarithm of the data is approximately
//! Normal.  It always produces right-skewed, positive-valued data — a hallmark of
//! many biological measurements.
//!
//! ## Medical applications
//!
//! | Biomarker / quantity | Why log-normal |
//! |----------------------|---------------|
//! | **CRP** (C-reactive protein, mg/L) | Spans < 1 to > 100 in the same cohort |
//! | **Serum creatinine** (µmol/L) | Positive, right-skewed in kidney disease |
//! | **Drug plasma concentration** (AUC) | Product of absorption / distribution factors |
//! | **Tumour volume** | Multiplicative growth process |
//! | **Hospital length-of-stay** | Most stays short, rare very long admissions |
//!
//! ## Example — CRP inflammation marker
//!
//! ```rust
//! use rs_stats::distributions::lognormal::LogNormal;
//! use rs_stats::distributions::traits::Distribution;
//!
//! // CRP levels (mg/L) — healthy < 5, elevated 5–100, critical > 100
//! let crp = vec![
//!     0.8, 1.2, 1.5, 2.1, 2.4, 3.2, 3.9, 5.6, 9.7, 12.4,
//!     22.3, 45.0, 88.0, 0.9, 1.3, 1.8, 0.7, 0.6, 0.5, 1.1,
//! ];
//! let dist = LogNormal::fit(&crp).unwrap();
//! println!("LogNormal(μ={:.2}, σ={:.2})", dist.mu, dist.sigma);
//!
//! // Median is more appropriate than mean for skewed biomarkers
//! let median = dist.inverse_cdf(0.5).unwrap();
//! let p_high = 1.0 - dist.cdf(10.0).unwrap();
//! println!("Median CRP       = {:.2} mg/L", median);
//! println!("P(CRP > 10 mg/L) = {:.1}%", p_high * 100.0);
//! ```

use std::f64::consts::PI;

use crate::distributions::normal_distribution::{normal_cdf, normal_inverse_cdf};
use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};

/// Log-Normal distribution LogNormal(μ, σ).
///
/// # Examples
/// ```
/// use rs_stats::distributions::lognormal::LogNormal;
/// use rs_stats::distributions::traits::Distribution;
///
/// let ln = LogNormal::new(0.0, 1.0).unwrap();
/// // Mean of LogNormal(0, 1) = exp(0.5)
/// assert!((ln.mean() - 0.5_f64.exp()).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LogNormal {
    /// Location (mean of ln X) μ
    pub mu: f64,
    /// Scale (std-dev of ln X) σ > 0
    pub sigma: f64,
}

impl LogNormal {
    /// Creates a `LogNormal(μ, σ)` distribution.
    pub fn new(mu: f64, sigma: f64) -> StatsResult<Self> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "LogNormal::new: sigma must be positive".to_string(),
            });
        }
        Ok(Self { mu, sigma })
    }

    /// Exact MLE: μ̂ = mean(ln data), σ̂ = population std-dev of ln data.
    ///
    /// Requires all data > 0.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "LogNormal::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x <= 0.0) {
            return Err(StatsError::InvalidInput {
                message: "LogNormal::fit: all data values must be positive".to_string(),
            });
        }
        let n = data.len() as f64;
        let log_data: Vec<f64> = data.iter().map(|&x| x.ln()).collect();
        let mu = log_data.iter().sum::<f64>() / n;
        let variance = log_data.iter().map(|&y| (y - mu).powi(2)).sum::<f64>() / n;
        Self::new(mu, variance.sqrt())
    }
}

impl Distribution for LogNormal {
    fn name(&self) -> &str {
        "LogNormal"
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
        let z = (x.ln() - self.mu) / self.sigma;
        Ok(-x.ln() - self.sigma.ln() - 0.5 * (2.0 * PI).ln() - 0.5 * z * z)
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        normal_cdf(x.ln(), self.mu, self.sigma)
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "LogNormal::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(normal_inverse_cdf(p, self.mu, self.sigma)?.exp())
    }

    fn mean(&self) -> f64 {
        (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }

    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (s2.exp() - 1.0) * (2.0 * self.mu + s2).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lognormal_mean() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert!((ln.mean() - 0.5_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal_pdf_positive() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        // pdf(1) = 1/(sqrt(2π)) ≈ 0.3989
        // INV_SQRT_2PI = 1/sqrt(2π) ≈ 0.3989422804014327
        assert!((ln.pdf(1.0).unwrap() - 0.398_942_280_4).abs() < 1e-8);
    }

    #[test]
    fn test_lognormal_cdf_bounds() {
        let ln = LogNormal::new(0.0, 1.0).unwrap();
        assert_eq!(ln.cdf(0.0).unwrap(), 0.0);
        assert!((ln.cdf(f64::MAX).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lognormal_inverse_cdf_roundtrip() {
        let ln = LogNormal::new(1.0, 0.5).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = ln.inverse_cdf(p).unwrap();
            let p_back = ln.cdf(x).unwrap();
            // Tolerance: Acklam approximation + erf roundtrip accumulates ~1e-6 error
            assert!((p - p_back).abs() < 1e-6, "p={p}: roundtrip failed");
        }
    }

    #[test]
    fn test_lognormal_fit() {
        let data = vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5, 1.2];
        let ln = LogNormal::fit(&data).unwrap();
        let log_mean = data.iter().map(|&x| x.ln()).sum::<f64>() / data.len() as f64;
        assert!((ln.mu - log_mean).abs() < 1e-10);
    }
}
