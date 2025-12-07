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

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;
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
pub struct BinomialConfig<T>
where
    T: ToPrimitive,
{
    /// The number of trials.
    pub n: u64,
    /// The probability of success in a single trial.
    pub p: T,
}

impl<T> BinomialConfig<T>
where
    T: ToPrimitive,
{
    /// Creates a new BinomialConfig with validation
    ///
    /// # Arguments
    /// * `n` - The number of trials
    /// * `p` - The probability of success
    ///
    /// # Returns
    /// `Some(BinomialConfig)` if parameters are valid, `None` otherwise
    pub fn new(n: u64, p: T) -> StatsResult<Self> {
        let p_64 = p.to_f64().ok_or_else(|| StatsError::ConversionError {
            message: "BinomialConfig::new: Failed to convert p to f64".to_string(),
        })?;

        if n == 0 {
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
/// # Errors
/// Returns an error if:
/// - n is zero
/// - p is not between 0 and 1
/// - k > n
/// - Type conversion to f64 fails
///
/// # Examples
/// ```
/// use rs_stats::distributions::binomial_distribution::pmf;
///
/// // Calculate probability of 3 successes in 10 trials with p=0.5
/// let prob = pmf(3, 10, 0.5).unwrap();
/// assert!((prob - 0.1171875).abs() < 1e-10);
/// ```
#[inline]
pub fn pmf<T>(k: u64, n: u64, p: T) -> StatsResult<f64>
where
    T: ToPrimitive,
{
    let p_64 = p.to_f64().ok_or_else(|| StatsError::ConversionError {
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
/// # Errors
/// Returns an error if:
/// - n is zero
/// - p is not between 0 and 1
/// - k > n
/// - Type conversion to f64 fails
///
/// # Examples
/// ```
/// use rs_stats::distributions::binomial_distribution::cdf;
///
/// // Calculate probability of 3 or fewer successes in 10 trials with p=0.5
/// let prob = cdf(3, 10, 0.5).unwrap();
/// assert!((prob - 0.171875).abs() < 1e-10);
/// ```
#[inline]
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
    (0..=k).try_fold(0.0, |acc, i| pmf(i, n, p).map(|prob| acc + prob))
}

/// Calculate the binomial coefficient (n choose k).
#[inline]
fn combination(n: u64, k: u64) -> StatsResult<f64> {
    if k > n {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::combination: k must be less than or equal to n"
                .to_string(),
        });
    }

    // Use a more numerically stable algorithm
    if k > n / 2 {
        return combination(n, n - k);
    }

    Ok((1..=k).fold(1.0_f64, |acc, i| acc * (n - i + 1) as f64 / i as f64))
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
    fn test_binomial_pmf_large_values_n() {
        // Test with values that exceed i32::MAX to verify overflow protection
        // Using values just above i32::MAX (2,147,483,647)
        let n = 2_200_000_000u64;
        let k = 5u64;
        let p = 0.5;

        // This should not panic or truncate - should use powf() path
        let result = pmf(k, n, p);

        // Result might be very small or NaN due to numerical precision, but shouldn't panic
        match result {
            Ok(val) => {
                // Value should be valid (might be very small due to large n)
                assert!(
                    !val.is_infinite(),
                    "PMF should not be infinite for large values"
                );
            }
            Err(_) => {
                // Error is acceptable for very large values (numerical precision limits)
            }
        }
    }

    #[test]
    fn test_binomial_pmf_large_values_k() {
        // Test with values that exceed i32::MAX to verify overflow protection
        // Using values just above i32::MAX (2,147,483,647)
        let n = 2u64;
        let k = 2_200_000_000_000u64;
        let p = 0.5;

        // This should not panic or truncate - should use powf() path
        let result = pmf(k, n, p);

        // Result might be very small or NaN due to numerical precision, but shouldn't panic
        match result {
            Ok(val) => {
                // Value should be valid (might be very small due to large n)
                assert!(
                    !val.is_infinite(),
                    "PMF should not be infinite for large values"
                );
            }
            Err(_) => {
                // Error is acceptable for very large values (numerical precision limits)
            }
        }
    }

    #[test]
    fn test_binomial_config_new_valid() {
        let config = BinomialConfig::new(10, 0.5);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.n, 10);
    }

    #[test]
    fn test_binomial_config_new_n_zero() {
        let result = BinomialConfig::new(0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_config_new_p_out_of_range_negative() {
        let result = BinomialConfig::new(10, -0.1);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_config_new_p_out_of_range_above_one() {
        let result = BinomialConfig::new(10, 1.1);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_config_new_p_zero() {
        let config = BinomialConfig::new(10, 0.0);
        assert!(config.is_ok());
    }

    #[test]
    fn test_binomial_config_new_p_one() {
        let config = BinomialConfig::new(10, 1.0);
        assert!(config.is_ok());
    }

    #[test]
    fn test_binomial_pmf_p_zero_k_zero() {
        // When p=0.0 and k=0, PMF should return combinations (which is 1 for k=0)
        let result = pmf(0, 10, 0.0).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_binomial_pmf_p_zero_k_greater_than_zero() {
        // When p=0.0 and k>0, PMF should return 0.0
        let result = pmf(5, 10, 0.0).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_binomial_pmf_p_one_k_equals_n() {
        // When p=1.0 and k=n, PMF should return combinations (which is 1 for k=n)
        let result = pmf(10, 10, 1.0).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_binomial_pmf_p_one_k_less_than_n() {
        // When p=1.0 and k<n, PMF should return 0.0
        let result = pmf(5, 10, 1.0).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_binomial_pmf_n_zero() {
        let result = pmf(0, 0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_pmf_p_out_of_range() {
        let result = pmf(5, 10, 1.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_cdf_k_greater_than_n() {
        let result = cdf(15, 10, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_combination_symmetry() {
        // Test that combination(n, k) == combination(n, n-k) when k > n/2
        // This tests the symmetry optimization path
        let n = 10u64;
        let k = 8u64; // k > n/2, so should use symmetry

        // Direct call should use symmetry path
        let result1 = combination(n, k).unwrap();
        // Should be same as combination(n, n-k)
        let result2 = combination(n, n - k).unwrap();
        assert_eq!(result1, result2);

        // Verify it's correct: C(10, 8) = C(10, 2) = 45
        assert_eq!(result1, 45.0);
    }

    #[test]
    fn test_binomial_combination_k_greater_than_n() {
        let result = combination(10, 15);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_combination_k_equals_n() {
        // C(n, n) = 1
        let result = combination(10, 10).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_binomial_combination_k_zero() {
        // C(n, 0) = 1
        let result = combination(10, 0).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_binomial_config_new_n_one() {
        // Test edge case: n = 1 (minimum valid value)
        let config = BinomialConfig::new(1, 0.5);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.n, 1);
    }

    #[test]
    fn test_binomial_pmf_k_greater_than_n() {
        // When k > n, combination() should return an error
        let result = pmf(15, 10, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_cdf_n_zero() {
        let result = cdf(5, 0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_cdf_p_out_of_range() {
        let result = cdf(5, 10, 1.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_combination_k_exactly_n_over_2() {
        // Test boundary case: k = n/2 (should not use symmetry)
        let n = 10u64;
        let k = 5u64; // k = n/2, should not use symmetry
        let result = combination(n, k).unwrap();
        // C(10, 5) = 252
        assert_eq!(result, 252.0);
    }

    #[test]
    fn test_binomial_combination_k_just_over_n_over_2() {
        // Test k = n/2 + 1 (should use symmetry)
        let n = 10u64;
        let k = 6u64; // k > n/2, should use symmetry
        let result1 = combination(n, k).unwrap();
        let result2 = combination(n, n - k).unwrap();
        assert_eq!(result1, result2);
        // C(10, 6) = C(10, 4) = 210
        assert_eq!(result1, 210.0);
    }
}
