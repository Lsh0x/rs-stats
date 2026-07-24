//! # Laplace Distribution
//!
//! The Laplace(μ, b) distribution (double exponential) is a symmetric,
//! sharply-peaked distribution with exponential tails — heavier than the
//! Normal but with all moments finite.
//!
//! **PDF**: f(x; μ, b) = exp(−|x − μ|/b) / (2b)
//!
//! **CDF**: F(x) = ½·exp((x − μ)/b) for x < μ, 1 − ½·exp(−(x − μ)/b) for x ≥ μ
//!
//! **Mean**: μ   **Variance**: 2b²
//!
//! ## Applications
//!
//! - Differential privacy (the Laplace mechanism)
//! - Robust regression (L1 loss ⇔ Laplace likelihood)
//! - Modeling errors with occasional large deviations
//!
//! ## Example
//!
//! ```rust
//! use rs_stats::distributions::laplace::Laplace;
//! use rs_stats::distributions::traits::Distribution;
//!
//! let l = Laplace::new(0.0, 1.0).unwrap();
//! assert!((l.cdf(0.0).unwrap() - 0.5).abs() < 1e-15);
//! assert!((l.variance() - 2.0).abs() < 1e-15);
//! ```

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use std::f64::consts::LN_2;

/// Laplace distribution Laplace(μ, b).
///
/// # Examples
/// ```
/// use rs_stats::distributions::laplace::Laplace;
/// use rs_stats::distributions::traits::Distribution;
///
/// let l = Laplace::new(1.0, 2.0).unwrap();
/// assert!((l.mean() - 1.0).abs() < 1e-15);
/// assert!((l.variance() - 8.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Laplace {
    /// Location parameter μ (mean, median and mode)
    pub mu: f64,
    /// Scale parameter b > 0 (diversity)
    pub b: f64,
}

impl Laplace {
    /// Creates a `Laplace(μ, b)` distribution.
    pub fn new(mu: f64, b: f64) -> StatsResult<Self> {
        if !mu.is_finite() || !b.is_finite() {
            return Err(StatsError::InvalidInput {
                message: "Laplace::new: parameters must be finite".to_string(),
            });
        }
        if b <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Laplace::new: b must be positive".to_string(),
            });
        }
        Ok(Self { mu, b })
    }

    /// Exact MLE: μ̂ = median(data), b̂ = mean(|xᵢ − μ̂|).
    ///
    /// The sample median maximises the likelihood in μ (L1 loss), and the
    /// mean absolute deviation from the median is the closed-form MLE of b.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Laplace::fit: data must not be empty".to_string(),
            });
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let mu = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        let b = data.iter().map(|&x| (x - mu).abs()).sum::<f64>() / n as f64;
        if b <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Laplace::fit: data has zero mean absolute deviation".to_string(),
            });
        }
        Self::new(mu, b)
    }
}

impl Distribution for Laplace {
    type X = f64;
    fn name(&self) -> &str {
        "Laplace"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        Ok(self.logpdf(x)?.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        Ok(-LN_2 - self.b.ln() - (x - self.mu).abs() / self.b)
    }

    fn log_likelihood(&self, data: &[f64]) -> StatsResult<f64> {
        Ok(self.log_likelihood_fast(data))
    }

    fn log_likelihood_fast(&self, data: &[f64]) -> f64 {
        // ln(2b) hoisted out of the loop — the per-point work is branch-free
        // (abs + multiply-add), so LLVM can autovectorise it.
        let log_norm = -LN_2 - self.b.ln();
        let inv_b = 1.0 / self.b;
        let mut acc = 0.0_f64;
        for &x in data {
            acc -= (x - self.mu).abs() * inv_b;
        }
        data.len() as f64 * log_norm + acc
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.mu) / self.b;
        // Exact exponential branches: the left tail ½·e^z never suffers the
        // 1 − (≈1) cancellation a single-formula CDF would.
        Ok(if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        })
    }

    /// Exact tail: `S(x) = ½·exp(−(x − μ)/b)` for `x ≥ μ` — full relative
    /// precision arbitrarily deep in the right tail.
    fn sf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.mu) / self.b;
        Ok(if z > 0.0 {
            0.5 * (-z).exp()
        } else {
            1.0 - 0.5 * z.exp()
        })
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Laplace::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Closed-form branches:
        //   p < ½ : x = μ + b·ln(2p)
        //   p ≥ ½ : x = μ − b·ln(2(1 − p)),  with ln(1 − p) via ln_1p(−p)
        //           so p's low bits survive for p → 1.
        Ok(if p < 0.5 {
            self.mu + self.b * (2.0 * p).ln()
        } else {
            self.mu - self.b * (LN_2 + (-p).ln_1p())
        })
    }

    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        2.0 * self.b * self.b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.laplace(loc=1, scale=2).
    const TOL: f64 = 1e-12;

    #[test]
    fn test_laplace_pdf_vs_scipy() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        assert!((l.pdf(3.0).unwrap() - 0.09196986029286058).abs() < TOL);
        assert!((l.logpdf(-0.5).unwrap() - (-2.136294361119891)).abs() < TOL);
        // Peak: pdf(μ) = 1/(2b)
        assert!((l.pdf(1.0).unwrap() - 0.25).abs() < TOL);
    }

    #[test]
    fn test_laplace_cdf_sf_vs_scipy() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        assert!((l.cdf(3.0).unwrap() - 0.8160602794142788).abs() < TOL);
        assert!((l.cdf(-0.5).unwrap() - 0.23618327637050734).abs() < TOL);
        assert!((l.sf(3.0).unwrap() - 0.18393972058572117).abs() < TOL);
        assert!((l.cdf(1.0).unwrap() - 0.5).abs() < TOL);
        assert!((l.sf(1.0).unwrap() - 0.5).abs() < TOL);
    }

    #[test]
    fn test_laplace_deep_tail_sf() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        // scipy: laplace(1, 2).sf(100) = 1.5899854500988748e-22
        let s = l.sf(100.0).unwrap();
        assert!(s > 0.0, "sf(100) underflowed");
        assert!((s / 1.5899854500988748e-22 - 1.0).abs() < 1e-12);
        // 1 − cdf would return exactly 0 here
        assert_eq!(1.0 - l.cdf(100.0).unwrap(), 0.0);
    }

    #[test]
    fn test_laplace_inverse_cdf_vs_scipy() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        assert!((l.inverse_cdf(0.2).unwrap() - (-0.83258146374831)).abs() < TOL);
        assert!((l.inverse_cdf(0.8).unwrap() - 2.8325814637483107).abs() < TOL);
        assert!((l.inverse_cdf(0.5).unwrap() - 1.0).abs() < TOL);
        assert_eq!(l.inverse_cdf(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(l.inverse_cdf(1.0).unwrap(), f64::INFINITY);
        assert!(l.inverse_cdf(-0.1).is_err());
        assert!(l.inverse_cdf(1.5).is_err());
    }

    #[test]
    fn test_laplace_roundtrip() {
        let l = Laplace::new(-2.0, 0.5).unwrap();
        for &x in &[-10.0, -2.5, -2.0, -1.9, 0.0, 3.0] {
            let p = l.cdf(x).unwrap();
            let x_back = l.inverse_cdf(p).unwrap();
            assert!((x - x_back).abs() < 1e-10 * x.abs().max(1.0), "x={x}");
        }
        for &p in &[1e-12, 0.1, 0.5, 0.9, 1.0 - 1e-12] {
            let x = l.inverse_cdf(p).unwrap();
            let p_back = l.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-13 * p.max(1e-3), "p={p}");
        }
    }

    #[test]
    fn test_laplace_fit_mle() {
        // Odd length: median = 2, MAD from median = (2+1+0+1+3)/5 = 1.4
        let data = vec![0.0, 1.0, 2.0, 3.0, 5.0];
        let l = Laplace::fit(&data).unwrap();
        assert!((l.mu - 2.0).abs() < 1e-12);
        assert!((l.b - 1.4).abs() < 1e-12);
        // Constant data: b = 0 → error
        assert!(Laplace::fit(&[7.0; 5]).is_err());
        assert!(Laplace::fit(&[]).is_err());
    }

    #[test]
    fn test_laplace_moments() {
        let l = Laplace::new(1.0, 2.0).unwrap();
        assert!((l.mean() - 1.0).abs() < TOL);
        assert!((l.variance() - 8.0).abs() < TOL);
    }

    #[test]
    fn test_laplace_log_likelihood_fast_matches_logpdf_sum() {
        let l = Laplace::new(0.5, 1.5).unwrap();
        let data = vec![-3.0, 0.0, 0.5, 2.0, 10.0];
        let expected: f64 = data.iter().map(|&x| l.logpdf(x).unwrap()).sum();
        assert!((l.log_likelihood_fast(&data) - expected).abs() < 1e-10);
        assert!((l.log_likelihood(&data).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_invalid_params() {
        assert!(Laplace::new(f64::NAN, 1.0).is_err());
        assert!(Laplace::new(0.0, f64::NAN).is_err());
        assert!(Laplace::new(f64::NEG_INFINITY, 1.0).is_err());
        assert!(Laplace::new(0.0, 0.0).is_err());
        assert!(Laplace::new(0.0, -2.0).is_err());
    }
}
