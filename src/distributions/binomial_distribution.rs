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
pub struct BinomialConfig {
    /// The number of trials.
    pub n: u64,
    /// The probability of success in a single trial.
    pub p: f64,
}

impl BinomialConfig {
    /// Creates a new BinomialConfig with validation
    ///
    /// # Arguments
    /// * `n` - The number of trials
    /// * `p` - The probability of success
    ///
    /// # Returns
    /// `Some(BinomialConfig)` if parameters are valid, `None` otherwise
    pub fn new(n: u64, p: f64) -> Option<Self> {
        if n > 0 && (0.0..=1.0).contains(&p) {
            Some(Self { n, p })
        } else {
            None
        }
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
/// let prob = pmf(3, 10, 0.5);
/// assert!((prob - 0.1171875).abs() < 1e-10);
/// ```
pub fn pmf(k: u64, n: u64, p: f64) -> f64 {
    assert!(n > 0, "n must be positive");
    assert!((0.0..=1.0).contains(&p), "p must be between 0 and 1");
    assert!(k <= n, "k must be less than or equal to n");

    let combinations = combination(n, k);
    let prob = (p.powf(k as f64)) * ((1.0 - p).powf((n - k) as f64));
    combinations * prob
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
/// let prob = cdf(3, 10, 0.5);
/// assert!((prob - 0.171875).abs() < 1e-10);
/// ```
pub fn cdf(k: u64, n: u64, p: f64) -> f64 {
    assert!(n > 0, "n must be positive");
    assert!((0.0..=1.0).contains(&p), "p must be between 0 and 1");
    assert!(k <= n, "k must be less than or equal to n");

    (0..=k).fold(0.0, |acc, i| acc + pmf(i, n, p))
}

/// Calculate the binomial coefficient (n choose k).
fn combination(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }

    // Use a more numerically stable algorithm
    if k > n / 2 {
        return combination(n, n - k);
    }

    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_pmf() {
        let n = 10;
        let p = 0.5;
        let k = 5;
        let result = pmf(k, n, p);
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
        let result = cdf(k, n, p);
        assert!(
            !result.is_nan(),
            "CDF returned NaN for k={}, n={}, p={}",
            k,
            n,
            p
        );
    }
}
