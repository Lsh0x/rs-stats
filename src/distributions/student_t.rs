//! # Student's t-Distribution
//!
//! The location-scale Student-t distribution with degrees of freedom ν,
//! location μ, and scale σ.
//!
//! **PDF**: f(x) = Γ((ν+1)/2) / (σ√(νπ) · Γ(ν/2)) · (1 + ((x−μ)/σ)² / ν)^(−(ν+1)/2)
//!
//! When σ=1 and μ=0 this is the standard t(ν).

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::{bisect_inverse_cdf, ln_gamma, regularized_incomplete_beta};
use std::f64::consts::PI;

/// Student's t-distribution with location μ, scale σ, and ν degrees of freedom.
///
/// # Examples
/// ```
/// use rs_stats::distributions::student_t::StudentT;
/// use rs_stats::distributions::traits::Distribution;
///
/// let t = StudentT::new(0.0, 1.0, 5.0).unwrap();
/// assert_eq!(t.mean(), 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    /// Location μ
    pub mu: f64,
    /// Scale σ > 0
    pub sigma: f64,
    /// Degrees of freedom ν > 0
    pub nu: f64,
}

impl StudentT {
    /// Creates a `StudentT(μ, σ, ν)` distribution.
    pub fn new(mu: f64, sigma: f64, nu: f64) -> StatsResult<Self> {
        if sigma <= 0.0 || nu <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "StudentT::new: sigma and nu must be positive".to_string(),
            });
        }
        Ok(Self { mu, sigma, nu })
    }

    /// MLE-like fitting via method of moments.
    ///
    /// - μ ≈ sample mean
    /// - σ ≈ sample std-dev
    /// - ν estimated from excess kurtosis: κ = 6/(ν−4) → ν = 4 + 6/κ
    ///   Falls back to ν = 30 (≈ Normal) when kurtosis cannot be estimated.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.len() < 4 {
            return Err(StatsError::InvalidInput {
                message: "StudentT::fit: at least 4 data points required".to_string(),
            });
        }
        let n = data.len() as f64;
        let mu = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n;
        let sigma = variance.sqrt();

        // Excess kurtosis: κ_4 = m4/σ⁴ - 3
        let m4 = data.iter().map(|&x| (x - mu).powi(4)).sum::<f64>() / n;
        let excess_kurtosis = m4 / (variance * variance) - 3.0;

        // ν = 4 + 6/κ  (from the identity κ_excess = 6/(ν−4), valid for ν > 4).
        // Threshold 0.01: below this the sample kurtosis is indistinguishable from 0
        // (Normal) given finite-sample noise, so we default to ν = 30 — a large enough
        // value that the t-distribution is virtually identical to Normal (< 0.1% diff).
        let nu = if excess_kurtosis > 0.01 {
            (4.0 + 6.0 / excess_kurtosis).max(2.01)
        } else {
            30.0 // ν ≥ 30 → t(ν) ≈ Normal; avoids artificially heavy tails from noisy kurtosis
        };

        Self::new(mu, sigma, nu)
    }

    /// Standard t CDF for the centred & scaled case.
    fn standard_t_cdf(&self, t: f64) -> f64 {
        let nu = self.nu;
        let x = nu / (t * t + nu);
        let ib = regularized_incomplete_beta(nu / 2.0, 0.5, x);
        if t >= 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
    }
}

impl Distribution for StudentT {
    fn name(&self) -> &str {
        "StudentT"
    }
    fn num_params(&self) -> usize {
        3
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        Ok(self.logpdf(x)?.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        let nu = self.nu;
        let z = (x - self.mu) / self.sigma;
        let log_coeff = ln_gamma((nu + 1.0) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * (nu * PI).ln()
            - self.sigma.ln();
        let log_tail = -((nu + 1.0) / 2.0) * (1.0 + z * z / nu).ln();
        Ok(log_coeff + log_tail)
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        let t = (x - self.mu) / self.sigma;
        Ok(self.standard_t_cdf(t))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "StudentT::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.5 {
            return Ok(self.mu);
        }
        // Bisection on the t-scale then transform back
        let std_dev_est = self.sigma * (self.nu / (self.nu - 2.0)).sqrt().max(1.0);
        let lo = self.mu - 30.0 * std_dev_est;
        let hi = self.mu + 30.0 * std_dev_est;
        let mu = self.mu;
        let sigma = self.sigma;
        let nu = self.nu;
        Ok(bisect_inverse_cdf(
            |x| {
                let t = (x - mu) / sigma;
                let ib = regularized_incomplete_beta(nu / 2.0, 0.5, nu / (t * t + nu));
                if t >= 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
            },
            p,
            lo,
            hi,
        ))
    }

    fn mean(&self) -> f64 {
        self.mu // defined for ν > 1; we return μ always
    }

    fn variance(&self) -> f64 {
        if self.nu > 2.0 {
            self.sigma * self.sigma * self.nu / (self.nu - 2.0)
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_student_t_pdf_symmetric() {
        let t = StudentT::new(0.0, 1.0, 5.0).unwrap();
        let p1 = t.pdf(1.0).unwrap();
        let p2 = t.pdf(-1.0).unwrap();
        assert!((p1 - p2).abs() < 1e-12);
    }

    #[test]
    fn test_student_t_cdf_symmetry() {
        let t = StudentT::new(0.0, 1.0, 5.0).unwrap();
        let c_pos = t.cdf(1.5).unwrap();
        let c_neg = t.cdf(-1.5).unwrap();
        assert!((c_pos + c_neg - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_student_t_cdf_median() {
        let t = StudentT::new(0.0, 1.0, 10.0).unwrap();
        assert!((t.cdf(0.0).unwrap() - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_student_t_inverse_cdf_roundtrip() {
        let t = StudentT::new(0.0, 1.0, 8.0).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = t.inverse_cdf(p).unwrap();
            let p_back = t.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn test_student_t_approaches_normal() {
        // t(ν→∞) → N(0,1): pdf at 0 should approach 1/sqrt(2π) ≈ 0.3989
        let t = StudentT::new(0.0, 1.0, 1000.0).unwrap();
        assert!((t.pdf(0.0).unwrap() - 0.398_942_280_4).abs() < 1e-4);
    }

    #[test]
    fn test_student_t_invalid() {
        assert!(StudentT::new(0.0, 0.0, 5.0).is_err());
        assert!(StudentT::new(0.0, 1.0, 0.0).is_err());
    }
}
