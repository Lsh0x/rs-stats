//! # Pareto Distribution (Type I)
//!
//! The Pareto(xₘ, α) distribution is the canonical power-law distribution:
//! "80/20 rule" phenomena where a small fraction of items carries most of the
//! total (wealth, city sizes, file sizes, insurance claims).
//!
//! **PDF**: f(x; xₘ, α) = α·xₘ^α / x^(α+1),  x ≥ xₘ
//!
//! **CDF**: F(x) = 1 − (xₘ/x)^α   **SF**: S(x) = (xₘ/x)^α (exact power-law tail)
//!
//! **Mean**: α·xₘ/(α − 1) for α > 1 (∞ otherwise)
//! **Variance**: xₘ²·α / [(α − 1)²(α − 2)] for α > 2 (∞ otherwise)
//!
//! ## Applications
//!
//! - Income and wealth modeling (Pareto's original use)
//! - Insurance: large-claim severity
//! - Network science: degree distributions, file-size distributions
//!
//! ## Example
//!
//! ```rust
//! use rs_stats::distributions::pareto::Pareto;
//! use rs_stats::distributions::traits::Distribution;
//!
//! let p = Pareto::new(2.0, 3.0).unwrap();
//! assert!((p.mean() - 3.0).abs() < 1e-12); // α·xₘ/(α−1) = 3·2/2
//! assert!((p.sf(5.0).unwrap() - 0.064).abs() < 1e-15); // (2/5)³
//! ```

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};

/// Pareto (type I) distribution Pareto(xₘ, α).
///
/// # Examples
/// ```
/// use rs_stats::distributions::pareto::Pareto;
/// use rs_stats::distributions::traits::Distribution;
///
/// let p = Pareto::new(1.0, 2.5).unwrap();
/// assert!((p.cdf(1.0).unwrap() - 0.0).abs() < 1e-15); // support starts at xₘ
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pareto {
    /// Scale parameter xₘ > 0 (minimum possible value)
    pub xm: f64,
    /// Shape parameter α > 0 (tail index)
    pub alpha: f64,
}

impl Pareto {
    /// Creates a `Pareto(xₘ, α)` distribution.
    pub fn new(xm: f64, alpha: f64) -> StatsResult<Self> {
        if !xm.is_finite() || !alpha.is_finite() {
            return Err(StatsError::InvalidInput {
                message: "Pareto::new: parameters must be finite".to_string(),
            });
        }
        if xm <= 0.0 || alpha <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Pareto::new: xm and alpha must be positive".to_string(),
            });
        }
        Ok(Self { xm, alpha })
    }

    /// Exact MLE: x̂ₘ = min(data), α̂ = n / Σ ln(xᵢ/x̂ₘ).
    ///
    /// Requires all data strictly positive. Errors if the data is constant
    /// (Σ ln(xᵢ/x̂ₘ) = 0 gives an infinite α̂).
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Pareto::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x <= 0.0) {
            return Err(StatsError::InvalidInput {
                message: "Pareto::fit: all data values must be positive".to_string(),
            });
        }
        let xm = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let sum_ln = data.iter().map(|&x| (x / xm).ln()).sum::<f64>();
        if sum_ln <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Pareto::fit: data is constant (alpha estimate diverges)".to_string(),
            });
        }
        Self::new(xm, data.len() as f64 / sum_ln)
    }
}

impl Distribution for Pareto {
    type X = f64;
    fn name(&self) -> &str {
        "Pareto"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        if x < self.xm {
            return Ok(0.0);
        }
        Ok(self.logpdf(x)?.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        if x < self.xm {
            return Ok(f64::NEG_INFINITY);
        }
        // ln f = ln α + α·ln xₘ − (α + 1)·ln x — stable arbitrarily far in
        // the tail where the direct power would underflow.
        Ok(self.alpha.ln() + self.alpha * self.xm.ln() - (self.alpha + 1.0) * x.ln())
    }

    fn log_likelihood(&self, data: &[f64]) -> StatsResult<f64> {
        Ok(self.log_likelihood_fast(data))
    }

    fn log_likelihood_fast(&self, data: &[f64]) -> f64 {
        // ln α + α·ln xₘ hoisted out of the loop; one ln per point remains.
        let log_norm = self.alpha.ln() + self.alpha * self.xm.ln();
        let a1 = self.alpha + 1.0;
        let mut acc = 0.0_f64;
        for &x in data {
            if x < self.xm {
                return f64::NEG_INFINITY;
            }
            acc -= a1 * x.ln();
        }
        data.len() as f64 * log_norm + acc
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= self.xm {
            return Ok(0.0);
        }
        // 1 − (xₘ/x)^α as −expm1(α·ln(xₘ/x)): keeps relative precision for
        // x barely above xₘ where the power is ≈ 1.
        Ok(-(self.alpha * (self.xm / x).ln()).exp_m1())
    }

    /// Exact power-law tail: `S(x) = (xₘ/x)^α` — full relative precision
    /// arbitrarily deep in the tail (never computed as 1 − cdf).
    fn sf(&self, x: f64) -> StatsResult<f64> {
        if x <= self.xm {
            return Ok(1.0);
        }
        Ok((self.xm / x).powf(self.alpha))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Pareto::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(self.xm);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Closed form x = xₘ·(1 − p)^(−1/α) = xₘ·exp(−ln(1 − p)/α), with
        // ln(1 − p) via ln_1p(−p) so p's low bits survive for p → 0.
        Ok(self.xm * (-(-p).ln_1p() / self.alpha).exp())
    }

    /// `α·xₘ/(α − 1)` for α > 1; `f64::INFINITY` otherwise.
    fn mean(&self) -> f64 {
        if self.alpha > 1.0 {
            self.alpha * self.xm / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// `xₘ²·α / [(α − 1)²(α − 2)]` for α > 2; `f64::INFINITY` otherwise.
    fn variance(&self) -> f64 {
        if self.alpha > 2.0 {
            let am1 = self.alpha - 1.0;
            self.xm * self.xm * self.alpha / (am1 * am1 * (self.alpha - 2.0))
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.pareto(b=3, scale=2)  (b = α, scale = xₘ).
    const TOL: f64 = 1e-12;

    #[test]
    fn test_pareto_pdf_vs_scipy() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        assert!((p.pdf(5.0).unwrap() - 0.038400000000000004).abs() < TOL);
        assert!((p.logpdf(5.0).unwrap() - (-3.259697819388456)).abs() < TOL);
        // At the support boundary: pdf(xₘ) = α/xₘ
        assert!((p.pdf(2.0).unwrap() - 1.5).abs() < TOL);
        // Below support
        assert_eq!(p.pdf(1.0).unwrap(), 0.0);
        assert_eq!(p.logpdf(1.0).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_pareto_cdf_sf_vs_scipy() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        assert!((p.cdf(5.0).unwrap() - 0.9359999999999999).abs() < TOL);
        assert!((p.sf(5.0).unwrap() - 0.064).abs() < TOL);
        assert_eq!(p.cdf(2.0).unwrap(), 0.0);
        assert_eq!(p.sf(2.0).unwrap(), 1.0);
        assert_eq!(p.cdf(0.5).unwrap(), 0.0);
        assert_eq!(p.sf(0.5).unwrap(), 1.0);
    }

    #[test]
    fn test_pareto_deep_tail_sf() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        // scipy: pareto(3, scale=2).sf(1e5) = 8e-15
        let s = p.sf(1e5).unwrap();
        assert!((s / 8e-15 - 1.0).abs() < 1e-12, "sf(1e5) = {s}");
    }

    #[test]
    fn test_pareto_inverse_cdf_vs_scipy() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        assert!((p.inverse_cdf(0.7).unwrap() - 2.987603164371443).abs() < TOL);
        assert_eq!(p.inverse_cdf(0.0).unwrap(), 2.0);
        assert_eq!(p.inverse_cdf(1.0).unwrap(), f64::INFINITY);
        assert!(p.inverse_cdf(-0.1).is_err());
        assert!(p.inverse_cdf(1.1).is_err());
    }

    #[test]
    fn test_pareto_roundtrip() {
        let p = Pareto::new(0.5, 1.7).unwrap();
        for &x in &[0.500001, 0.6, 1.0, 10.0, 1e4] {
            let q = p.cdf(x).unwrap();
            let x_back = p.inverse_cdf(q).unwrap();
            assert!((x - x_back).abs() < 1e-9 * x, "x={x}");
        }
        for &q in &[1e-12, 0.1, 0.5, 0.9, 1.0 - 1e-12] {
            let x = p.inverse_cdf(q).unwrap();
            let q_back = p.cdf(x).unwrap();
            assert!((q - q_back).abs() < 1e-12 * q.max(1e-3), "q={q}");
        }
    }

    #[test]
    fn test_pareto_moments_vs_scipy() {
        let p = Pareto::new(2.0, 3.0).unwrap();
        // scipy: pareto(3, scale=2).mean() = 3.0, .var() = 3.0
        assert!((p.mean() - 3.0).abs() < TOL);
        assert!((p.variance() - 3.0).abs() < TOL);
        // Heavy-tail regimes
        assert_eq!(Pareto::new(2.0, 1.0).unwrap().mean(), f64::INFINITY);
        assert_eq!(Pareto::new(2.0, 0.5).unwrap().mean(), f64::INFINITY);
        assert_eq!(Pareto::new(2.0, 2.0).unwrap().variance(), f64::INFINITY);
        assert_eq!(Pareto::new(2.0, 1.5).unwrap().variance(), f64::INFINITY);
        // α = 1.5 > 1: mean finite even though variance is not
        assert!(Pareto::new(2.0, 1.5).unwrap().mean().is_finite());
    }

    #[test]
    fn test_pareto_fit_mle() {
        // xₘ = min = 1.0; α̂ = n / Σ ln(xᵢ/1)
        let data = vec![1.0, 2.0, 4.0, 8.0];
        let p = Pareto::fit(&data).unwrap();
        assert!((p.xm - 1.0).abs() < TOL);
        let expected_alpha = 4.0 / (2.0_f64.ln() + 4.0_f64.ln() + 8.0_f64.ln());
        assert!((p.alpha - expected_alpha).abs() < TOL);
        // Invalid data
        assert!(Pareto::fit(&[]).is_err());
        assert!(Pareto::fit(&[1.0, -2.0, 3.0]).is_err());
        assert!(Pareto::fit(&[1.0, 0.0, 3.0]).is_err());
        // Constant data → Σ ln = 0 → error
        assert!(Pareto::fit(&[3.0; 6]).is_err());
    }

    #[test]
    fn test_pareto_log_likelihood_fast_matches_logpdf_sum() {
        let p = Pareto::new(1.5, 2.2).unwrap();
        let data = vec![1.6, 2.0, 3.7, 10.0, 55.0];
        let expected: f64 = data.iter().map(|&x| p.logpdf(x).unwrap()).sum();
        assert!((p.log_likelihood_fast(&data) - expected).abs() < 1e-10);
        assert!((p.log_likelihood(&data).unwrap() - expected).abs() < 1e-10);
        // Out-of-support point → −∞
        assert_eq!(p.log_likelihood_fast(&[1.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_pareto_invalid_params() {
        assert!(Pareto::new(f64::NAN, 1.0).is_err());
        assert!(Pareto::new(1.0, f64::NAN).is_err());
        assert!(Pareto::new(f64::INFINITY, 1.0).is_err());
        assert!(Pareto::new(0.0, 1.0).is_err());
        assert!(Pareto::new(-1.0, 1.0).is_err());
        assert!(Pareto::new(1.0, 0.0).is_err());
        assert!(Pareto::new(1.0, -3.0).is_err());
    }
}
