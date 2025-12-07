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

use num_traits::ToPrimitive;
use crate::error::{StatsResult, StatsError};
use serde::{Deserialize, Serialize};

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
pub struct PoissonConfig<T> where T: ToPrimitive {
    /// The average rate (λ) of events.
    pub lambda: T,
}

impl<T> PoissonConfig<T> where T: ToPrimitive {
    /// Creates a new PoissonConfig with validation
    ///
    /// # Arguments
    /// * `lambda` - The average rate (λ) of events
    ///
    /// # Returns
    /// `Ok(PoissonConfig)` if lambda is positive, `Err` otherwise
    pub fn new(lambda: T) -> StatsResult<Self>  {
        let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
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
/// # Panics
/// Panics if lambda is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::poisson_distribution::pmf;
///
/// // Calculate probability of 2 events with λ=1.5
/// let prob = pmf(2, 1.5).unwrap();
/// assert!((prob - 0.2510214301669835).abs() < 1e-10);
/// ```
pub fn pmf<T>(k: u64, lambda: T) -> StatsResult<f64> where T: ToPrimitive {
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
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
    
    let fact = (1..=k as usize).fold(1.0, |acc, x| acc * x as f64);
    
    // Use log-space calculation to avoid:
    // 1. Casting u64 to i32 (information loss)
    // 2. Numerical underflow/overflow with large exponents
    // 3. Better numerical stability
    // Formula: λ^k * e^(-λ) / k! = exp(k * ln(λ) - λ - ln(k!))
    let k_f64 = k as f64;
    
    // Calculate in log space: k * ln(λ) - λ - ln(k!)
    // Note: ln(k!) = sum(ln(i)) for i=1..=k, but we already computed k! above
    let log_lambda_power = k_f64 * lambda_64.ln();
    let log_prob = log_lambda_power - lambda_64 - fact.ln();
    
    // Convert back from log space
    let prob = log_prob.exp();
    
    Ok(prob)
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
/// # Panics
/// Panics if lambda is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::poisson_distribution::cdf;
///
/// // Calculate probability of 2 or fewer events with λ=1.5
/// let prob = cdf(2, 1.5).unwrap();
/// assert!((prob - 0.8088468305380586).abs() < 1e-10);
/// ```
pub fn cdf<T>(k: u64, lambda: T) -> StatsResult<f64> where T: ToPrimitive {
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "poisson_distribution::cdf: Failed to convert lambda to f64".to_string(),
    })?;
    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "poisson_distribution::cdf: lambda must be positive".to_string(),
        });
    }
    (0..=k).try_fold(0.0, |acc, i| pmf(i, lambda_64).map(|prob| acc + prob))
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
}
