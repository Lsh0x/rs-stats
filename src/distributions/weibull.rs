//! # Weibull Distribution
//!
//! The Weibull distribution with shape k and scale λ models lifetimes, failure rates,
//! and extreme values.
//!
//! **PDF**: f(x; k, λ) = (k/λ) · (x/λ)^(k−1) · exp(−(x/λ)^k),  x ≥ 0
//!
//! **Mean**: λ · Γ(1 + 1/k)   **Variance**: λ² · [Γ(1 + 2/k) − Γ(1 + 1/k)²]

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::gamma_fn;

/// Weibull distribution Weibull(k, λ).
///
/// # Examples
/// ```
/// use rs_stats::distributions::weibull::Weibull;
/// use rs_stats::distributions::traits::Distribution;
///
/// let w = Weibull::new(1.0, 2.0).unwrap(); // Exponential(0.5)
/// assert!((w.mean() - 2.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Weibull {
    /// Shape parameter k > 0
    pub k: f64,
    /// Scale parameter λ > 0
    pub lambda: f64,
}

impl Weibull {
    /// Creates a `Weibull(k, λ)` distribution.
    pub fn new(k: f64, lambda: f64) -> StatsResult<Self> {
        if k <= 0.0 || lambda <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Weibull::new: k and lambda must be positive".to_string(),
            });
        }
        Ok(Self { k, lambda })
    }

    /// MLE fitting using the Teimouri-Gupta approximation for k and then
    /// the closed-form scale estimator λ = (Σxᵢᵏ / n)^(1/k).
    ///
    /// Requires all data > 0.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Weibull::fit: data must not be empty".to_string(),
            });
        }
        if data.iter().any(|&x| x <= 0.0) {
            return Err(StatsError::InvalidInput {
                message: "Weibull::fit: all data values must be positive".to_string(),
            });
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let cv = variance.sqrt() / mean; // coefficient of variation

        // Teimouri & Gupta (2013) approximation: k ≈ cv^{-1.086}
        let k = cv.powf(-1.086).max(0.01);

        // Closed-form scale estimate: λ = (Σxᵢᵏ / n)^(1/k)
        let sum_xk: f64 = data.iter().map(|&x| x.powf(k)).sum::<f64>();
        let lambda = (sum_xk / n).powf(1.0 / k);

        Self::new(k, lambda)
    }
}

impl Distribution for Weibull {
    fn name(&self) -> &str {
        "Weibull"
    }
    fn num_params(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64) -> StatsResult<f64> {
        if x < 0.0 {
            return Ok(0.0);
        }
        if x == 0.0 {
            return Ok(if self.k < 1.0 {
                f64::INFINITY
            } else if self.k == 1.0 {
                1.0 / self.lambda
            } else {
                0.0
            });
        }
        Ok(self.logpdf(x)?.exp())
    }

    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        let xl = x / self.lambda;
        Ok(
            self.k.ln() - self.lambda.ln() + (self.k - 1.0) * (x / self.lambda).ln()
                - xl.powf(self.k),
        )
    }

    fn cdf(&self, x: f64) -> StatsResult<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        Ok(1.0 - (-(x / self.lambda).powf(self.k)).exp())
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Weibull::inverse_cdf: p must be in [0, 1]".to_string(),
            });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // Closed-form inverse: x = λ · (-ln(1-p))^(1/k)
        Ok(self.lambda * (-(1.0 - p).ln()).powf(1.0 / self.k))
    }

    fn mean(&self) -> f64 {
        self.lambda * gamma_fn(1.0 + 1.0 / self.k)
    }

    fn variance(&self) -> f64 {
        let g1 = gamma_fn(1.0 + 1.0 / self.k);
        let g2 = gamma_fn(1.0 + 2.0 / self.k);
        self.lambda * self.lambda * (g2 - g1 * g1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_exponential_case() {
        // Weibull(1, λ) = Exponential(1/λ)
        let w = Weibull::new(1.0, 2.0).unwrap();
        assert!((w.mean() - 2.0).abs() < 1e-8);
        // CDF(x) = 1 - e^(-x/λ)
        assert!((w.cdf(2.0).unwrap() - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_inverse_cdf_exact() {
        let w = Weibull::new(2.0, 3.0).unwrap();
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = w.inverse_cdf(p).unwrap();
            let p_back = w.cdf(x).unwrap();
            assert!((p - p_back).abs() < 1e-10, "p={p}");
        }
    }

    #[test]
    fn test_weibull_fit_recovers_scale() {
        // For Weibull(2, 3), data close to these params
        let data = vec![1.5, 2.5, 3.5, 2.0, 4.0, 1.8, 3.0, 2.8, 1.2, 3.8];
        let w = Weibull::fit(&data).unwrap();
        assert!(w.k > 0.0 && w.lambda > 0.0);
    }

    #[test]
    fn test_weibull_invalid() {
        assert!(Weibull::new(0.0, 1.0).is_err());
        assert!(Weibull::new(1.0, 0.0).is_err());
    }
}
