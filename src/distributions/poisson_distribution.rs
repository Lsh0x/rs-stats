//! # Poisson Distribution
//!
//! This module implements the Poisson distribution, a discrete probability distribution
//! that models the probability of a given number of events occurring in a fixed interval
//! of time or space.
//!
//! ## Key Characteristics
//! - Parameterized by λ (lambda), the average rate of occurrence
//! - Discrete probability distribution
//! - Models rare events in large populations
//!
//! ## Common Applications
//! - Modeling call center traffic
//! - Predicting system failures
//! - Analyzing radioactive decay
//! - Counting website visitors
//!
//! ## Mathematical Formulation
//! The probability mass function (PMF) is given by:
//!
//! P(X = k) = (e^(-λ) * λ^k) / k!
//!
//! where:
//! - λ is the expected number of occurrences
//! - k is the number of occurrences
//! - e is Euler's number (~2.71828)

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

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

/// Configuration for the Poisson distribution.
///
/// # Fields
/// * `lambda` - The average rate (λ) of events. Must be positive.
///
/// # Examples
/// ```
/// use rs_stats::distributions::poisson_distribution::PoissonConfig;
///
/// let config = PoissonConfig { lambda: 2.5 };
/// assert!(config.lambda > 0.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PoissonConfig<T>
where
    T: ToPrimitive,
{
    /// The average rate (λ) of events.
    pub lambda: T,
}

impl<T> PoissonConfig<T>
where
    T: ToPrimitive,
{
    /// Creates a new PoissonConfig with validation
    ///
    /// # Arguments
    /// * `lambda` - The average rate (λ) of events
    ///
    /// # Returns
    /// `Ok(PoissonConfig)` if lambda is positive, `Err` otherwise
    pub fn new(lambda: T) -> StatsResult<Self> {
        let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError {
            message: "PoissonConfig::new: Failed to convert lambda to f64".to_string(),
        })?;

        if lambda_64 > 0.0 {
            Ok(Self { lambda })
        } else {
            Err(StatsError::InvalidInput {
                message: "PoissonConfig::new: lambda must be positive".to_string(),
            })
        }
    }
}

/// Probability mass function (PMF) for the Poisson distribution.
///
/// Calculates the probability of observing exactly `k` events given the rate `λ`.
///
/// # Arguments
/// * `k` - The number of occurrences (must be a non-negative integer)
/// * `lambda` - The average rate (λ) (must be positive)
///
/// # Returns
/// The probability of exactly `k` events occurring.
///
/// # Errors
/// Returns an error if:
/// - lambda is not positive
/// - Type conversion to f64 fails
/// - k is too large to compute factorial
///
/// # Examples
/// ```
/// use rs_stats::distributions::poisson_distribution::pmf;
///
/// // Calculate probability of 2 events with λ=1.5
/// let prob = pmf(2, 1.5).unwrap();
/// assert!((prob - 0.2510214301669835).abs() < 1e-10);
/// ```
#[inline]
pub fn pmf<T>(k: u64, lambda: T) -> StatsResult<f64>
where
    T: ToPrimitive,
{
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "poisson_distribution::pmf: Failed to convert lambda to f64".to_string(),
    })?;
    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "poisson_distribution::pmf: lambda must be positive".to_string(),
        });
    }
    // Check if k fits in usize (for factorial calculation)
    if k > usize::MAX as u64 {
        return Err(StatsError::InvalidInput {
            message: "poisson_distribution::pmf: k is too large to compute factorial".to_string(),
        });
    }

    // Use log-space calculation: exp(k * ln(λ) - λ - ln(k!))
    // ln(k!) computed in O(1) via Stirling approximation (with exact lookup for small k)
    let k_f64 = k as f64;
    let log_lambda_power = k_f64 * lambda_64.ln();
    let log_fact = ln_factorial(k);
    let log_prob = log_lambda_power - lambda_64 - log_fact;

    Ok(log_prob.exp())
}

/// Cumulative distribution function (CDF) for the Poisson distribution.
///
/// Calculates the probability of observing `k` or fewer events given the rate `λ`.
///
/// # Arguments
/// * `k` - The maximum number of occurrences (must be a non-negative integer)
/// * `lambda` - The average rate (λ) (must be positive)
///
/// # Returns
/// The cumulative probability of `k` or fewer events occurring.
///
/// # Errors
/// Returns an error if:
/// - lambda is not positive
/// - Type conversion to f64 fails
/// - k is too large to compute factorial
///
/// # Examples
/// ```
/// use rs_stats::distributions::poisson_distribution::cdf;
///
/// // Calculate probability of 2 or fewer events with λ=1.5
/// let prob = cdf(2, 1.5).unwrap();
/// assert!((prob - 0.8088468305380586).abs() < 1e-10);
/// ```
#[inline]
pub fn cdf<T>(k: u64, lambda: T) -> StatsResult<f64>
where
    T: ToPrimitive,
{
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "poisson_distribution::cdf: Failed to convert lambda to f64".to_string(),
    })?;
    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "poisson_distribution::cdf: lambda must be positive".to_string(),
        });
    }
    // Incremental log-factorial computation: O(k) total, O(1) per step.
    // Each step reuses the previous log_fact via addition.
    let ln_lambda = lambda_64.ln();
    let mut log_fact = 0.0_f64;
    let mut cdf_sum = 0.0_f64;
    for i in 0..=k {
        if i > 0 {
            log_fact += (i as f64).ln();
        }
        let log_pmf = (i as f64) * ln_lambda - lambda_64 - log_fact;
        cdf_sum += log_pmf.exp();
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
