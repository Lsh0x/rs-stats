//! # Binomial Distribution
//!
//! This module implements the Binomial distribution, a discrete probability distribution
//! that models the number of successes in a sequence of independent experiments.
//!
//! ## Key Characteristics
//! - Models the number of successes in `n` independent trials
//! - Each trial has success probability `p`
//! - Discrete probability distribution
//!
//! ## Common Applications
//! - Quality control testing
//! - A/B testing
//! - Risk analysis
//! - Genetics (Mendelian inheritance)
//!
//! ## Mathematical Formulation
//! The probability mass function (PMF) is given by:
//!
//! P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
//!
//! where:
//! - n is the number of trials
//! - k is the number of successes
//! - p is the probability of success
//! - C(n,k) is the binomial coefficient (n choose k)

use num_traits::ToPrimitive;
use crate::error::{StatsResult, StatsError};
use serde::{Deserialize, Serialize};

/// Configuration for the Binomial distribution.
///
/// # Fields
/// * `n` - The number of trials (must be positive)
/// * `p` - The probability of success (must be between 0 and 1)
///
/// # Examples
/// ```
/// use rs_stats::distributions::binomial_distribution::BinomialConfig;
///
/// let config = BinomialConfig { n: 10, p: 0.5 };
/// assert!(config.n > 0);
/// assert!(config.p >= 0.0 && config.p <= 1.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BinomialConfig<T> where T: ToPrimitive {
    /// The number of trials.
    pub n: u64,
    /// The probability of success in a single trial.
    pub p: T,
}

impl<T> BinomialConfig<T> where T: ToPrimitive {
    /// Creates a new BinomialConfig with validation
    ///
    /// # Arguments
    /// * `n` - The number of trials
    /// * `p` - The probability of success
    ///
    /// # Returns
    /// `Some(BinomialConfig)` if parameters are valid, `None` otherwise
    pub fn new(n: u64, p: T) -> StatsResult<Self> {
        let p_64 = p.to_f64().ok_or_else(|| StatsError::ConversionError{
            message: "BinomialConfig::new: Failed to convert p to f64".to_string(),
        })?;

        if n <= 0 {
            return Err(StatsError::InvalidInput {
                message: "BinomialConfig::new: n must be positive".to_string(),
            });
        }
        if !((0.0..=1.0).contains(&p_64)) {
            return Err(StatsError::InvalidInput {
                message: "BinomialConfig::new: p must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { n, p })
    }
}

/// Probability mass function (PMF) for the Binomial distribution.
///
/// Calculates the probability of observing exactly `k` successes in `n` trials
/// with success probability `p`.
///
/// # Arguments
/// * `k` - The number of successes (must be ≤ n)
/// * `n` - The total number of trials (must be positive)
/// * `p` - The probability of success in a single trial (must be between 0 and 1)
///
/// # Returns
/// The probability of exactly `k` successes occurring.
///
/// # Panics
/// Panics if:
/// - n is zero
/// - p is not between 0 and 1
/// - k > n
///
/// # Examples
/// ```
/// use rs_stats::distributions::binomial_distribution::pmf;
///
/// // Calculate probability of 3 successes in 10 trials with p=0.5
/// let prob = pmf(3, 10, 0.5).unwrap();
/// assert!((prob - 0.1171875).abs() < 1e-10);
/// ```
pub fn pmf<T>(k: u64, n: u64, p: T) -> StatsResult<f64> where T: ToPrimitive {
    let p_64 = p.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "binomial_distribution::pmf: Failed to convert p to f64".to_string(),
    })?;
    if n == 0 {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::pmf: n must be positive".to_string(),
        });
    }
    if !((0.0..=1.0).contains(&p_64)) {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::pmf: p must be between 0 and 1".to_string(),
        });
    }
    let combinations = combination(n, k)?;
    
    // Use log-space calculation to avoid:
    // 1. Casting u64 to i32 (information loss)
    // 2. Numerical underflow/overflow with large exponents
    // 3. Better numerical stability
    // Formula: p^k * (1-p)^(n-k) = exp(k * ln(p) + (n-k) * ln(1-p))
    
    // Handle edge cases explicitly for correctness
    if p_64 == 0.0 {
        // If p = 0, then p^k = 0 for k > 0, and 1 for k = 0
        return Ok(if k == 0 { combinations } else { 0.0 });
    }
    if p_64 == 1.0 {
        // If p = 1, then (1-p)^(n-k) = 0 for k < n, and 1 for k = n
        return Ok(if k == n { combinations } else { 0.0 });
    }
    
    // Convert to f64 (no information loss for reasonable values)
    let k_f64 = k as f64;
    let n_minus_k_f64 = (n - k) as f64;
    
    // Calculate in log space: k * ln(p) + (n-k) * ln(1-p)
    // Both p and (1-p) are guaranteed to be in (0, 1) here
    let log_prob = k_f64 * p_64.ln() + n_minus_k_f64 * (1.0 - p_64).ln();
    
    // Convert back from log space
    let prob = log_prob.exp();
    
    Ok(combinations * prob)
}

/// Cumulative distribution function (CDF) for the Binomial distribution.
///
/// Calculates the probability of observing `k` or fewer successes in `n` trials
/// with success probability `p`.
///
/// # Arguments
/// * `k` - The maximum number of successes (must be ≤ n)
/// * `n` - The total number of trials (must be positive)
/// * `p` - The probability of success in a single trial (must be between 0 and 1)
///
/// # Returns
/// The cumulative probability of `k` or fewer successes occurring.
///
/// # Panics
/// Panics if:
/// - n is zero
/// - p is not between 0 and 1
/// - k > n
///
/// # Examples
/// ```
/// use rs_stats::distributions::binomial_distribution::cdf;
///
/// // Calculate probability of 3 or fewer successes in 10 trials with p=0.5
/// let prob = cdf(3, 10, 0.5).unwrap();
/// assert!((prob - 0.171875).abs() < 1e-10);
/// ```
pub fn cdf(k: u64, n: u64, p: f64) -> StatsResult<f64> {
    if n == 0 {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::cdf: n must be positive".to_string(),
        });
    }
    if !((0.0..=1.0).contains(&p)) {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::cdf: p must be between 0 and 1".to_string(),
        });
    }
    if k > n {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::cdf: k must be less than or equal to n".to_string(),
        });
    }
    (0..=k).try_fold(0.0, |acc, i| {
        pmf(i, n, p).map(|prob| acc + prob)
    })
}

/// Calculate the binomial coefficient (n choose k).
fn combination(n: u64, k: u64) -> StatsResult<f64> {

    if k > n {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::combination: k must be less than or equal to n".to_string(),
        });
    }

    // Use a more numerically stable algorithm
    if k > n / 2 {
        return combination(n, n - k);
    }

    Ok((1..=k).fold(1.0 as f64, |acc, i| acc * (n - i + 1) as f64 / i as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_pmf() {
        let n = 10;
        let p = 0.5;
        let k = 5;
        let result = pmf(k, n, p).unwrap();
        assert!(
            !result.is_nan(),
            "PMF returned NaN for k={}, n={}, p={}",
            k,
            n,
            p
        );
    }

    #[test]
    fn test_binomial_cdf() {
        let n = 10;
        let p = 0.5;
        let k = 5;
        let result = cdf(k, n, p).unwrap();
        assert!(
            !result.is_nan(),
            "CDF returned NaN for k={}, n={}, p={}",
            k,
            n,
            p
        );
    }

    #[test]
    fn test_binomial_pmf_large_values() {
        // Test with values that exceed i32::MAX to verify overflow protection
        // Using values just above i32::MAX (2,147,483,647)
        let n = 2_500_000_000u64; // 2.5 billion
        let k = 1_250_000_000u64; // 1.25 billion (half of n)
        let p = 0.5;
        
        // This should not panic or truncate - should use powf() path
        let result = pmf(k, n, p);
        
        // Result might be very small or NaN due to numerical precision, but shouldn't panic
        match result {
            Ok(val) => {
                // Value should be valid (might be very small due to large n)
                assert!(!val.is_infinite(), "PMF should not be infinite for large values");
            }
            Err(_) => {
                // Error is acceptable for very large values (numerical precision limits)
            }
        }
    }
}
