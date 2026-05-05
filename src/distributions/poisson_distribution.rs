//! # Poisson Distribution
//!
//! Models the number of independent events that occur in a fixed interval of time
//! or space when events happen at a constant average rate λ.
//!
//! **PMF**: P(X = k) = e^(−λ) · λ^k / k!,  k = 0, 1, 2, …
//!
//! **Mean** = **Variance** = λ
//!
//! ## When to use
//!
//! Use Poisson when:
//! - Events are independent (one event does not affect the probability of another)
//! - The rate λ is constant over the observation window
//! - The number of events in non-overlapping windows are independent
//!
//! ## Medical applications
//!
//! | Scenario | λ interpretation |
//! |----------|-----------------|
//! | Hospital-acquired infections | infections per ward per month |
//! | Adverse drug reactions | events per 1 000 prescriptions |
//! | Emergency department arrivals | patients per hour |
//! | Surgical site infections | cases per 100 procedures |
//! | Mutations in a tumour | mutations per cell division |
//!
//! ## Example
//!
//! ```rust
//! use rs_stats::distributions::poisson_distribution::Poisson;
//! use rs_stats::DiscreteDistribution;
//!
//! // Hospital-acquired infections (HAI): historical rate λ = 2.3 per ward/month
//! let hai = Poisson::new(2.3).unwrap();
//!
//! // Probability of a zero-infection month (baseline benchmark)
//! println!("P(0 HAI)  = {:.1}%", hai.pmf(0).unwrap() * 100.0);   // ≈ 10.0%
//!
//! // Alert threshold: P(≥ 5 infections) — trigger infection-control review
//! let p_alert = 1.0 - hai.cdf(4).unwrap();
//! println!("P(≥5 HAI) = {:.1}%", p_alert * 100.0);
//!
//! // Fit from 12 months of observed counts (MLE: λ̂ = sample mean)
//! let monthly = vec![1.0, 3.0, 2.0, 0.0, 4.0, 2.0, 1.0, 3.0, 2.0, 1.0, 5.0, 2.0];
//! let fitted = Poisson::fit(&monthly).unwrap();
//! println!("Estimated λ = {:.2} infections/month", fitted.lambda);
//! ```

use crate::error::{StatsError, StatsResult};

/// Precomputed ln(k!) for k = 0..=20 (exact values).
/// For k > 20, we use the Stirling approximation: ln(k!) ≈ k*ln(k) - k + 0.5*ln(2πk).
/// This makes PMF O(1) per call instead of O(k).
const LN_FACT_TABLE: [f64; 21] = [
    0.0,                     // ln(0!) = 0
    0.0,                     // ln(1!) = 0
    std::f64::consts::LN_2,  // ln(2!) = ln(2)
    1.791_759_469_228_055,   // ln(3!)
    3.178_053_830_347_945_7, // ln(4!)
    4.787_491_742_782_046,   // ln(5!)
    6.579_251_212_010_101,   // ln(6!)
    8.525_161_361_065_415,   // ln(7!)
    10.604_602_902_745_25,   // ln(8!)
    12.801_827_480_081_469,  // ln(9!)
    15.104_412_573_075_516,  // ln(10!)
    17.502_307_845_873_887,  // ln(11!)
    19.987_214_495_661_885,  // ln(12!)
    22.552_163_853_123_42,   // ln(13!)
    25.191_221_182_738_68,   // ln(14!)
    27.899_271_383_840_89,   // ln(15!)
    30.671_860_128_909_07,   // ln(16!)
    33.505_073_450_136_89,   // ln(17!)
    36.395_445_208_033_05,   // ln(18!)
    39.339_884_187_199_49,   // ln(19!)
    42.335_616_460_753_485,  // ln(20!)
];

/// Compute ln(k!) in O(1) time.
/// Uses a precomputed table for k <= 20 and Stirling's approximation for k > 20.
#[inline]
fn ln_factorial(k: u64) -> f64 {
    if k <= 20 {
        LN_FACT_TABLE[k as usize]
    } else {
        // Stirling's approximation: ln(k!) ≈ k*ln(k) - k + 0.5*ln(2πk)
        let k_f64 = k as f64;
        k_f64 * k_f64.ln() - k_f64 + 0.5 * (2.0 * std::f64::consts::PI * k_f64).ln()
    }
}

// Private math helpers; the public API is the [`Poisson`] struct's
// [`DiscreteDistribution`] impl below.

#[inline]
fn pmf(k: u64, lambda: f64) -> StatsResult<f64> {
    if lambda <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "poisson::pmf: lambda must be positive".to_string(),
        });
    }
    let k_f64 = k as f64;
    let log_prob = k_f64 * lambda.ln() - lambda - ln_factorial(k);
    Ok(log_prob.exp())
}

#[inline]
fn cdf(k: u64, lambda: f64) -> StatsResult<f64> {
    if lambda <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "poisson::cdf: lambda must be positive".to_string(),
        });
    }
    // Incremental log-factorial: O(k) total, O(1) per step.
    let ln_lambda = lambda.ln();
    let mut log_fact = 0.0_f64;
    let mut cdf_sum = 0.0_f64;
    for i in 0..=k {
        if i > 0 {
            log_fact += (i as f64).ln();
        }
        cdf_sum += ((i as f64) * ln_lambda - lambda - log_fact).exp();
    }
    Ok(cdf_sum.clamp(0.0, 1.0))
}

// ── Typed struct + DiscreteDistribution impl ───────────────────────────────────

/// Poisson distribution Poisson(λ) as a typed struct.
///
/// # Examples
/// ```
/// use rs_stats::distributions::poisson_distribution::Poisson;
/// use rs_stats::distributions::traits::DiscreteDistribution;
///
/// let p = Poisson::new(3.0).unwrap();
/// assert!((p.mean() - 3.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    /// Rate parameter λ > 0
    pub lambda: f64,
}

impl Poisson {
    /// Creates a `Poisson` distribution with validation.
    pub fn new(lambda: f64) -> StatsResult<Self> {
        if lambda <= 0.0 {
            return Err(StatsError::InvalidInput {
                message: "Poisson::new: lambda must be positive".to_string(),
            });
        }
        Ok(Self { lambda })
    }

    /// MLE: λ = mean(data).
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Poisson::fit: data must not be empty".to_string(),
            });
        }
        let lambda = data.iter().sum::<f64>() / data.len() as f64;
        Self::new(lambda)
    }
}

impl crate::distributions::traits::DiscreteDistribution for Poisson {
    fn name(&self) -> &str {
        "Poisson"
    }
    fn num_params(&self) -> usize {
        1
    }
    fn pmf(&self, k: u64) -> StatsResult<f64> {
        pmf(k, self.lambda)
    }
    fn logpmf(&self, k: u64) -> StatsResult<f64> {
        // ln P(k) = k*ln(λ) - λ - ln(k!)
        let ln_fact = ln_factorial(k);
        Ok((k as f64) * self.lambda.ln() - self.lambda - ln_fact)
    }
    fn cdf(&self, k: u64) -> StatsResult<f64> {
        cdf(k, self.lambda)
    }
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_pmf() {
        let lambda = 2.0;
        let k = 0;
        let result = pmf(k, lambda).unwrap();
        assert!(
            !result.is_nan(),
            "PMF returned NaN for k={}, lambda={}",
            k,
            lambda
        );
    }

    #[test]
    fn test_poisson_cdf() {
        let lambda = 2.0;
        let k = 5;
        let result = cdf(k, lambda);
        assert!(
            !result.unwrap().is_nan(),
            "CDF returned NaN for k={}, lambda={}",
            k,
            lambda
        );
    }

    #[test]
    fn test_poisson_pmf_k_zero() {
        // When k = 0, PMF should be e^(-λ)
        let lambda: f64 = 2.0;
        let k = 0;
        let result = pmf(k, lambda).unwrap();
        let expected = (-lambda).exp();
        assert!(
            (result - expected).abs() < 1e-10,
            "PMF for k=0 should be e^(-λ)"
        );
    }

    #[test]
    fn test_poisson_pmf_k_greater_than_zero() {
        // When k > 0, PMF should use the full formula
        let lambda: f64 = 2.0;
        let k = 3;
        let result = pmf(k, lambda).unwrap();
        // Expected: λ^k * e^(-λ) / k! = 2^3 * e^(-2) / 6 = 8 * e^(-2) / 6
        let expected =
            (lambda.powi(k as i32) * (-lambda).exp()) / (1..=k as usize).product::<usize>() as f64;
        assert!(
            (result - expected).abs() < 1e-10,
            "PMF for k>0 should match formula"
        );
    }

    #[test]
    fn test_poisson_pmf_lambda_zero() {
        let result = pmf(0, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_poisson_pmf_lambda_negative() {
        let result = pmf(0, -1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_poisson_cdf_k_zero() {
        // CDF at k=0 should equal PMF at k=0
        let lambda = 2.0;
        let k = 0;
        let cdf_result = cdf(k, lambda).unwrap();
        let pmf_result = pmf(k, lambda).unwrap();
        assert!(
            (cdf_result - pmf_result).abs() < 1e-10,
            "CDF at k=0 should equal PMF at k=0"
        );
    }

    #[test]
    fn test_poisson_pmf_k_too_large() {
        // Test k > usize::MAX branch
        // Use a value that's definitely larger than usize::MAX
        let k = if usize::MAX as u64 == u64::MAX {
            // On platforms where usize is u64, we can't test this branch
            // So we skip this test
            return;
        } else {
            usize::MAX as u64 + 1
        };
        let lambda = 2.0;
        let result = pmf(k, lambda);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_poisson_cdf_lambda_zero() {
        let result = cdf(5, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_poisson_cdf_lambda_negative() {
        let result = cdf(5, -1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }
}
