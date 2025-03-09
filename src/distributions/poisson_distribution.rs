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

use serde::{Serialize, Deserialize};

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
pub struct PoissonConfig {
    /// The average rate (λ) of events.
    pub lambda: f64,
}

impl PoissonConfig {
    /// Creates a new PoissonConfig with validation
    ///
    /// # Arguments
    /// * `lambda` - The average rate (λ) of events
    ///
    /// # Returns
    /// `Some(PoissonConfig)` if lambda is positive, `None` otherwise
    pub fn new(lambda: f64) -> Option<Self> {
        if lambda > 0.0 {
            Some(Self { lambda })
        } else {
            None
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
/// let prob = pmf(2, 1.5);
/// assert!((prob - 0.2510214301669835).abs() < 1e-10);
/// ```
pub fn pmf(k: u64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    let e = std::f64::consts::E;
    let fact = (1..=k as usize).fold(1.0, |acc, x| acc * x as f64);
    ((e.powf(-lambda)) * (lambda.powf(k as f64))) / fact
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
/// let prob = cdf(2, 1.5);
/// assert!((prob - 0.8088468305380586).abs() < 1e-10);
/// ```
pub fn cdf(k: u64, lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    (0..=k).fold(0.0, |acc, i| acc + pmf(i, lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_pmf() {
        let lambda = 2.0;
        let k = 0;
        let result = pmf(k, lambda);
        assert!(!result.is_nan(), "PMF returned NaN for k={}, lambda={}", k, lambda);
    }

    #[test]
    fn test_poisson_cdf() {
        let lambda = 2.0;
        let k = 5;
        let result = cdf(k, lambda);
        assert!(!result.is_nan(), "CDF returned NaN for k={}, lambda={}", k, lambda);
    }
}