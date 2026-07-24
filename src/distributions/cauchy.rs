//! # Cauchy Distribution
//!
//! The Cauchy(x₀, γ) distribution (also known as the Lorentz distribution) is a
//! heavy-tailed continuous distribution. It arises as the ratio of two
//! independent standard normal variables and describes resonance line shapes
//! in physics.
//!
//! **PDF**: f(x; x₀, γ) = 1 / (πγ · [1 + ((x − x₀)/γ)²])
//!
//! **CDF**: F(x) = 1/2 + arctan((x − x₀)/γ) / π  (closed-form, invertible)
//!
//! **Mean / Variance**: *undefined* — the tails are so heavy that no moment of
//! order ≥ 1 exists. [`Distribution::mean`] and [`Distribution::variance`]
//! return `f64::NAN` for this distribution.
//!
//! ## Applications
//!
//! - Spectroscopy (Lorentzian line profiles)
//! - Robust statistics benchmarks (sample mean never converges)
//! - Modeling extreme financial returns
//!
//! ## Example
//!
//! ```rust
//! use rs_stats::distributions::cauchy::Cauchy;
//! use rs_stats::distributions::traits::Distribution;
//!
//! let c = Cauchy::new(0.0, 1.0).unwrap(); // standard Cauchy
//! assert!((c.cdf(0.0).unwrap() - 0.5).abs() < 1e-15);
//! assert!((c.inverse_cdf(0.75).unwrap() - 1.0).abs() < 1e-12);
//! assert!(c.mean().is_nan()); // undefined
//! ```

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use std::f64::consts::PI;

/// Cauchy distribution Cauchy(x₀, γ).
///
/// # Examples
/// ```
/// use rs_stats::distributions::cauchy::Cauchy;
/// use rs_stats::distributions::traits::Distribution;
///
/// let c = Cauchy::new(2.0, 1.5).unwrap();
/// assert!((c.inverse_cdf(0.5).unwrap() - 2.0).abs() < 1e-15); // median = x₀
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cauchy {
    /// Location parameter x₀ (median and mode)
    pub x0: f64,
    /// Scale parameter γ > 0 (half-width at half-maximum)
    pub gamma: f64,
}

impl Cauchy {
    /// Creates a `Cauchy(x₀, γ)` distribution.
    pub fn new(x0: f64, gamma: f64) -> StatsResult<Self> {
        if !x0.is_finite() || !gamma.is_finite() {
            return Err(StatsError::InvalidInput {
                message: "Cauchy::new: parameters must be finite".to_string(),
            });
        }
        if gamma <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Cauchy::new: gamma must be positive".to_string(),
            });
        }
        Ok(Self { x0, gamma })
    }

    /// Robust fitting: sample median for x₀ and half the interquartile
    /// range (IQR / 2) for γ.
    ///
    /// MLE has no closed form for the Cauchy, and moment estimators do not
    /// exist (the distribution has no mean). The median and IQR/2 are the
    /// standard consistent quantile estimators: for a Cauchy the theoretical
    /// quartiles are x₀ ± γ, so (Q₃ − Q₁)/2 = γ.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Cauchy::fit: data must not be empty".to_string(),
            });
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let x0 = quantile_sorted(&sorted, 0.5);
        let gamma = (quantile_sorted(&sorted, 0.75) - quantile_sorted(&sorted, 0.25)) / 2.0;
        if gamma <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Cauchy::fit: data has zero interquartile range".to_string(),
            });
        }
        Self::new(x0, gamma)
    }
}

/// Linear-interpolation quantile on pre-sorted data (numpy default method).
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

impl Distribution for Cauchy {
    type X = f64;
    fn name(&self) -> &str {
        "Cauchy"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.x0) / self.gamma;
        Ok(1.0 / (PI * self.gamma * (1.0 + z * z)))
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.x0) / self.gamma;
        // ln_1p(z²) keeps precision for |z| ≪ 1 where 1 + z² rounds to 1.
        Ok(-(PI * self.gamma).ln() - (z * z).ln_1p())
    }

    fn log_likelihood(&self, data: &[f64]) -> StatsResult<f64> {
        Ok(self.log_likelihood_fast(data))
    }

    fn log_likelihood_fast(&self, data: &[f64]) -> f64 {
        // ln(πγ) hoisted out of the per-point loop (one ln_1p per point remains).
        let log_norm = -(PI * self.gamma).ln();
        let inv_gamma = 1.0 / self.gamma;
        let mut acc = 0.0_f64;
        for &x in data {
            let z = (x - self.x0) * inv_gamma;
            acc -= (z * z).ln_1p();
        }
        data.len() as f64 * log_norm + acc
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.x0) / self.gamma;
        // Left tail via the identity 1/2 + atan(z)/π = atan(−1/z)/π for
        // z < 0: avoids the catastrophic cancellation of 0.5 − (≈0.5).
        Ok(if z < 0.0 {
            (-1.0 / z).atan() / PI
        } else {
            0.5 + z.atan() / PI
        })
    }

    /// Exact tail: `S(x) = atan(γ/(x − x₀))/π` for `x > x₀` — full relative
    /// precision deep into the right tail (no `0.5 − 0.5` cancellation).
    fn sf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.x0) / self.gamma;
        Ok(if z > 0.0 {
            (1.0 / z).atan() / PI
        } else {
            0.5 - z.atan() / PI
        })
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Cauchy::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Closed form x₀ + γ·tan(π(p − ½)), evaluated per branch through the
        // cotangent identity tan(π(p − ½)) = −1/tan(πp): computing p − ½
        // directly would drop p's low bits for p → 0 and p → 1.
        Ok(if p == 0.5 {
            self.x0
        } else if p < 0.5 {
            self.x0 - self.gamma / (PI * p).tan()
        } else {
            // 1 − p is exact for p ∈ [0.5, 1] (Sterbenz lemma).
            self.x0 + self.gamma / (PI * (1.0 - p)).tan()
        })
    }

    /// The Cauchy distribution has **no mean** (the defining integral
    /// diverges); returns `f64::NAN`.
    fn mean(&self) -> f64 {
        f64::NAN
    }

    /// The Cauchy distribution has **no variance** (no moments of order ≥ 1
    /// exist); returns `f64::NAN`.
    fn variance(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.cauchy(loc=2, scale=1.5).
    const TOL: f64 = 1e-12;

    #[test]
    fn test_cauchy_pdf_vs_scipy() {
        let c = Cauchy::new(2.0, 1.5).unwrap();
        assert!((c.pdf(4.0).unwrap() - 0.07639437268410977).abs() < TOL);
        assert!((c.logpdf(4.0).unwrap() - (-2.5718462414895455)).abs() < TOL);
        // Mode: pdf(x0) = 1/(πγ)
        assert!((c.pdf(2.0).unwrap() - 1.0 / (PI * 1.5)).abs() < TOL);
    }

    #[test]
    fn test_cauchy_cdf_sf_vs_scipy() {
        let c = Cauchy::new(2.0, 1.5).unwrap();
        assert!((c.cdf(4.0).unwrap() - 0.7951672353008665).abs() < TOL);
        assert!((c.sf(4.0).unwrap() - 0.20483276469913345).abs() < TOL);
        assert!((c.cdf(-1.0).unwrap() - 0.14758361765043326).abs() < TOL);
        assert!((c.cdf(2.0).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn test_cauchy_deep_tail_sf() {
        let c = Cauchy::new(2.0, 1.5).unwrap();
        // scipy: cauchy(2, 1.5).sf(1e6) = 4.774657842068963e-07
        let s = c.sf(1e6).unwrap();
        assert!(
            (s - 4.774657842068963e-07).abs() < 1e-19 * 1e12, // relative ~1e-12
            "sf(1e6) = {s}"
        );
        assert!((s / 4.774657842068963e-07 - 1.0).abs() < 1e-12);
        // Symmetric left tail via cdf
        let l = c.cdf(-1e6).unwrap();
        assert!(l > 0.0 && l < 1e-6);
    }

    #[test]
    fn test_cauchy_inverse_cdf_vs_scipy() {
        let c = Cauchy::new(2.0, 1.5).unwrap();
        assert!((c.inverse_cdf(0.3).unwrap() - 0.9101862079919583).abs() < TOL);
        assert!((c.inverse_cdf(0.9).unwrap() - 6.616525305762882).abs() < TOL);
        assert!((c.inverse_cdf(0.5).unwrap() - 2.0).abs() < TOL);
        assert_eq!(c.inverse_cdf(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(c.inverse_cdf(1.0).unwrap(), f64::INFINITY);
        assert!(c.inverse_cdf(-0.1).is_err());
        assert!(c.inverse_cdf(1.1).is_err());
    }

    #[test]
    fn test_cauchy_roundtrip() {
        let c = Cauchy::new(-3.0, 0.7).unwrap();
        for &x in &[-10.0, -3.5, -3.0, -2.9, 0.0, 5.0, 100.0] {
            let p = c.cdf(x).unwrap();
            let x_back = c.inverse_cdf(p).unwrap();
            assert!((x - x_back).abs() < 1e-9 * x.abs().max(1.0), "x={x}");
        }
        for &p in &[1e-10, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-10] {
            let x = c.inverse_cdf(p).unwrap();
            let p_back = c.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-12 * p.max(1e-3), "p={p}");
        }
    }

    #[test]
    fn test_cauchy_moments_undefined() {
        let c = Cauchy::new(2.0, 1.5).unwrap();
        assert!(c.mean().is_nan());
        assert!(c.variance().is_nan());
        assert!(c.std_dev().is_nan());
    }

    #[test]
    fn test_cauchy_fit_quantile_estimators() {
        // Odd-length data with known median / quartiles.
        let data = vec![-4.0, -1.0, 0.0, 1.0, 4.0];
        let c = Cauchy::fit(&data).unwrap();
        assert!((c.x0 - 0.0).abs() < 1e-12);
        // Q1 = -1, Q3 = 1 → gamma = 1
        assert!((c.gamma - 1.0).abs() < 1e-12);
        // Constant data: zero IQR → error
        assert!(Cauchy::fit(&[3.0; 8]).is_err());
        assert!(Cauchy::fit(&[]).is_err());
    }

    #[test]
    fn test_cauchy_log_likelihood_fast_matches_logpdf_sum() {
        let c = Cauchy::new(1.0, 2.0).unwrap();
        let data = vec![-5.0, -0.5, 1.0, 2.5, 40.0];
        let expected: f64 = data.iter().map(|&x| c.logpdf(x).unwrap()).sum();
        assert!((c.log_likelihood_fast(&data) - expected).abs() < 1e-10);
        assert!((c.log_likelihood(&data).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_invalid_params() {
        assert!(Cauchy::new(f64::NAN, 1.0).is_err());
        assert!(Cauchy::new(0.0, f64::NAN).is_err());
        assert!(Cauchy::new(f64::INFINITY, 1.0).is_err());
        assert!(Cauchy::new(0.0, 0.0).is_err());
        assert!(Cauchy::new(0.0, -1.5).is_err());
    }
}
