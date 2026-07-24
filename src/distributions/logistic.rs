//! # Logistic Distribution
//!
//! The Logistic(μ, s) distribution looks like a Normal with slightly heavier
//! tails; its CDF is the logistic (sigmoid) function, which makes it the
//! natural noise model behind logistic regression.
//!
//! **PDF**: f(x; μ, s) = e^(−z) / (s·(1 + e^(−z))²),  z = (x − μ)/s
//!
//! **CDF**: F(x) = 1 / (1 + e^(−z))   **SF**: S(x) = 1 / (1 + e^(z)) (exact tail)
//!
//! **Mean**: μ   **Variance**: s²π²/3
//!
//! ## Applications
//!
//! - Logistic regression / discrete choice models (latent-utility noise)
//! - Growth curves (population, adoption)
//! - Item response theory (Rasch models)
//!
//! ## Example
//!
//! ```rust
//! use rs_stats::distributions::logistic::Logistic;
//! use rs_stats::distributions::traits::Distribution;
//!
//! let l = Logistic::new(0.0, 1.0).unwrap();
//! assert!((l.cdf(0.0).unwrap() - 0.5).abs() < 1e-15);
//! assert!((l.variance() - std::f64::consts::PI.powi(2) / 3.0).abs() < 1e-12);
//! ```

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use std::f64::consts::PI;

/// Logistic distribution Logistic(μ, s).
///
/// # Examples
/// ```
/// use rs_stats::distributions::logistic::Logistic;
/// use rs_stats::distributions::traits::Distribution;
///
/// let l = Logistic::new(2.0, 1.5).unwrap();
/// assert!((l.mean() - 2.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Logistic {
    /// Location parameter μ (mean, median and mode)
    pub mu: f64,
    /// Scale parameter s > 0
    pub s: f64,
}

impl Logistic {
    /// Creates a `Logistic(μ, s)` distribution.
    pub fn new(mu: f64, s: f64) -> StatsResult<Self> {
        if !mu.is_finite() || !s.is_finite() {
            return Err(StatsError::InvalidInput {
                message: "Logistic::new: parameters must be finite".to_string(),
            });
        }
        if s <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Logistic::new: s must be positive".to_string(),
            });
        }
        Ok(Self { mu, s })
    }

    /// Method-of-moments fitting: μ̂ = mean(data), ŝ = √3·σ̂/π where σ̂ is
    /// the population standard deviation (Var = s²π²/3 ⇒ s = σ√3/π).
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Logistic::fit: data must not be empty".to_string(),
            });
        }
        // Single-pass online mean+variance (Welford).
        let mut count = 0.0_f64;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for &x in data {
            count += 1.0;
            let delta = x - mean;
            mean += delta / count;
            m2 += delta * (x - mean);
        }
        let std_pop = (m2 / count).sqrt();
        if std_pop <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Logistic::fit: data has zero variance".to_string(),
            });
        }
        Self::new(mean, 3.0_f64.sqrt() * std_pop / PI)
    }
}

impl Distribution for Logistic {
    type X = f64;
    fn name(&self) -> &str {
        "Logistic"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        Ok(self.logpdf(x)?.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        // Symmetric stable form: ln f = −|z| − 2·ln(1 + e^(−|z|)) − ln s.
        // e^(−|z|) never overflows and ln_1p keeps precision in both tails.
        let az = ((x - self.mu) / self.s).abs();
        Ok(-az - 2.0 * (-az).exp().ln_1p() - self.s.ln())
    }

    fn log_likelihood(&self, data: &[f64]) -> StatsResult<f64> {
        Ok(self.log_likelihood_fast(data))
    }

    fn log_likelihood_fast(&self, data: &[f64]) -> f64 {
        // ln s hoisted out of the per-point loop.
        let log_norm = -self.s.ln();
        let inv_s = 1.0 / self.s;
        let mut acc = 0.0_f64;
        for &x in data {
            let az = ((x - self.mu) * inv_s).abs();
            acc -= az + 2.0 * (-az).exp().ln_1p();
        }
        data.len() as f64 * log_norm + acc
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.mu) / self.s;
        // 1/(1 + e^(−z)): exact in the left tail (→ e^z), saturates
        // harmlessly to 1 in the right tail (use `sf` there).
        Ok(1.0 / (1.0 + (-z).exp()))
    }

    /// Exact tail: `S(x) = 1/(1 + exp(z))` — for large `z` this evaluates to
    /// `≈ e^(−z)` with full relative precision (never computed as 1 − cdf).
    fn sf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.mu) / self.s;
        Ok(1.0 / (1.0 + z.exp()))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Logistic::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Closed-form logit: x = μ + s·ln(p/(1 − p)) = μ + s·(ln p − ln(1 − p)),
        // with ln(1 − p) via ln_1p(−p) so p's low bits survive for p → 1.
        Ok(self.mu + self.s * (p.ln() - (-p).ln_1p()))
    }

    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.s * self.s * PI * PI / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.logistic(loc=2, scale=1.5).
    const TOL: f64 = 1e-12;

    #[test]
    fn test_logistic_pdf_vs_scipy() {
        let l = Logistic::new(2.0, 1.5).unwrap();
        assert!((l.pdf(4.0).unwrap() - 0.11006067310193583).abs() < TOL);
        assert!((l.logpdf(4.0).unwrap() - (-2.206723491596594)).abs() < TOL);
        // Peak: pdf(μ) = 1/(4s)
        assert!((l.pdf(2.0).unwrap() - 1.0 / 6.0).abs() < TOL);
    }

    #[test]
    fn test_logistic_cdf_sf_vs_scipy() {
        let l = Logistic::new(2.0, 1.5).unwrap();
        assert!((l.cdf(4.0).unwrap() - 0.791391472673955).abs() < TOL);
        assert!((l.sf(4.0).unwrap() - 0.20860852732604496).abs() < TOL);
        assert!((l.cdf(2.0).unwrap() - 0.5).abs() < TOL);
        assert!((l.sf(2.0).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn test_logistic_deep_tail_sf() {
        let l = Logistic::new(2.0, 1.5).unwrap();
        // scipy: logistic(2, 1.5).sf(50) = 1.2664165549094016e-14 ≈ e^(−48/1.5)
        let s = l.sf(50.0).unwrap();
        assert!(s > 0.0, "sf(50) underflowed");
        assert!((s / 1.2664165549094016e-14 - 1.0).abs() < 1e-12);
        // 1 − cdf loses most significant digits here (sf keeps them all)
        let naive = 1.0 - l.cdf(50.0).unwrap();
        assert!((naive / 1.2664165549094016e-14 - 1.0).abs() > 1e-8);
        // Deep left tail via cdf ≈ e^z stays exact too
        let left = l.cdf(-50.0).unwrap();
        assert!(left > 0.0 && left < 1e-14);
    }

    #[test]
    fn test_logistic_inverse_cdf_vs_scipy() {
        let l = Logistic::new(2.0, 1.5).unwrap();
        assert!((l.inverse_cdf(0.3).unwrap() - 0.7290532094191944).abs() < TOL);
        assert!((l.inverse_cdf(0.999).unwrap() - 12.360132167972829).abs() < TOL);
        assert!((l.inverse_cdf(0.5).unwrap() - 2.0).abs() < TOL);
        assert_eq!(l.inverse_cdf(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(l.inverse_cdf(1.0).unwrap(), f64::INFINITY);
        assert!(l.inverse_cdf(-0.2).is_err());
        assert!(l.inverse_cdf(1.2).is_err());
    }

    #[test]
    fn test_logistic_roundtrip() {
        let l = Logistic::new(-1.0, 0.4).unwrap();
        for &x in &[-8.0, -1.5, -1.0, -0.9, 0.0, 6.0] {
            let p = l.cdf(x).unwrap();
            let x_back = l.inverse_cdf(p).unwrap();
            // x → p → x through the saturating region of the CDF (p → 1) is
            // inherently ill-conditioned; 1e-7 absolute is the honest bound.
            assert!((x - x_back).abs() < 1e-7 * x.abs().max(1.0), "x={x}");
        }
        for &p in &[1e-12, 0.1, 0.5, 0.9, 1.0 - 1e-12] {
            let x = l.inverse_cdf(p).unwrap();
            let p_back = l.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-12 * p.max(1e-3), "p={p}");
        }
    }

    #[test]
    fn test_logistic_moments_vs_scipy() {
        let l = Logistic::new(2.0, 1.5).unwrap();
        assert!((l.mean() - 2.0).abs() < TOL);
        // scipy: logistic(2, 1.5).var() = 7.4022033008170185
        assert!((l.variance() - 7.4022033008170185).abs() < TOL);
    }

    #[test]
    fn test_logistic_fit_mom() {
        // mean = 3, population variance = 2 → s = √3·√2/π
        let data = vec![1.0, 3.0, 5.0, 3.0, 2.0, 4.0];
        let l = Logistic::fit(&data).unwrap();
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        assert!((l.mu - mean).abs() < 1e-12);
        assert!((l.s - 3.0_f64.sqrt() * var.sqrt() / PI).abs() < 1e-12);
        // Constant data: zero variance → error
        assert!(Logistic::fit(&[4.0; 5]).is_err());
        assert!(Logistic::fit(&[]).is_err());
    }

    #[test]
    fn test_logistic_log_likelihood_fast_matches_logpdf_sum() {
        let l = Logistic::new(0.3, 2.5).unwrap();
        let data = vec![-20.0, -1.0, 0.3, 4.0, 30.0];
        let expected: f64 = data.iter().map(|&x| l.logpdf(x).unwrap()).sum();
        assert!((l.log_likelihood_fast(&data) - expected).abs() < 1e-10);
        assert!((l.log_likelihood(&data).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logistic_invalid_params() {
        assert!(Logistic::new(f64::NAN, 1.0).is_err());
        assert!(Logistic::new(0.0, f64::NAN).is_err());
        assert!(Logistic::new(f64::INFINITY, 1.0).is_err());
        assert!(Logistic::new(0.0, 0.0).is_err());
        assert!(Logistic::new(0.0, -1.0).is_err());
    }
}
