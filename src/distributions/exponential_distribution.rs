//! # Exponential Distribution
//!
//! This module implements the exponential distribution, a continuous probability distribution
//! that models the time between events in a Poisson point process.
//!
//! ## Key Characteristics
//! - Continuous probability distribution
//! - Memoryless property (future states depend only on the present, not the past)
//! - Used to model waiting times and inter-arrival times
//!
//! ## Common Applications
//! - Time between arrivals in a Poisson process
//! - Lifetime analysis (e.g., how long until a component fails)
//! - Queue theory and service times
//! - Radioactive decay
//!
//! ## Mathematical Formulation
//! The probability density function (PDF) is given by:
//!
//! f(x; λ) = λe^(-λx) for x ≥ 0
//!
//! where:
//! - λ (lambda) is the rate parameter (λ > 0)
//! - x is the random variable (time or space)
//!
//! The cumulative distribution function (CDF) is:
//!
//! F(x; λ) = 1 - e^(-λx) for x ≥ 0

use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::Exp;
use serde::{Serialize, Deserialize};

/// Configuration for the Exponential distribution.
///
/// # Fields
/// * `lambda` - The rate parameter (must be positive)
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::ExponentialConfig;
///
/// let config = ExponentialConfig { lambda: 2.0 };
/// assert!(config.lambda > 0.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExponentialConfig {
    /// The rate parameter.
    pub lambda: f64,
}

impl ExponentialConfig {
    /// Creates a new ExponentialConfig with validation
    ///
    /// # Arguments
    /// * `lambda` - The rate parameter (must be positive)
    ///
    /// # Returns
    /// `Some(ExponentialConfig)` if parameter is valid, `None` otherwise
    pub fn new(lambda: f64) -> Option<Self> {
        if lambda > 0.0 {
            Some(Self { lambda })
        } else {
            None
        }
    }
}

/// Probability density function (PDF) for the Exponential distribution.
///
/// Calculates the probability density at point `x` for an exponential distribution
/// with rate parameter `lambda`.
///
/// # Arguments
/// * `x` - The point at which to evaluate the PDF (must be non-negative)
/// * `lambda` - The rate parameter (must be positive)
///
/// # Returns
/// The probability density at point `x`.
///
/// # Panics
/// Panics if:
/// - `x` is negative
/// - `lambda` is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::exponential_pdf;
///
/// // Calculate PDF at x = 1.0 with rate parameter lambda = 2.0
/// let pdf = exponential_pdf(1.0, 2.0);
/// assert!((pdf - 0.27067).abs() < 1e-5);
/// ```
pub fn exponential_pdf(x: f64, lambda: f64) -> f64 {
    assert!(x >= 0.0, "x must be non-negative");
    assert!(lambda > 0.0, "lambda must be positive");
    
    if x == 0.0 {
        lambda
    } else {
        lambda * (-lambda * x).exp()
    }
}

/// Cumulative distribution function (CDF) for the Exponential distribution.
///
/// Calculates the probability of a random variable being less than or equal to `x`
/// for an exponential distribution with rate parameter `lambda`.
///
/// # Arguments
/// * `x` - The point at which to evaluate the CDF (must be non-negative)
/// * `lambda` - The rate parameter (must be positive)
///
/// # Returns
/// The cumulative probability at point `x`.
///
/// # Panics
/// Panics if:
/// - `x` is negative
/// - `lambda` is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::exponential_cdf;
///
/// // Calculate CDF at x = 1.0 with rate parameter lambda = 2.0
/// let cdf = exponential_cdf(1.0, 2.0);
/// assert!((cdf - 0.86466).abs() < 1e-5);
/// ```
pub fn exponential_cdf(x: f64, lambda: f64) -> f64 {
    assert!(x >= 0.0, "x must be non-negative");
    assert!(lambda > 0.0, "lambda must be positive");
    
    1.0 - (-lambda * x).exp()
}

/// Inverse cumulative distribution function for the Exponential distribution.
///
/// Calculates the value of `x` for which the CDF equals the given probability `p`.
///
/// # Arguments
/// * `p` - The probability (must be between 0 and 1)
/// * `lambda` - The rate parameter (must be positive)
///
/// # Returns
/// The value `x` such that P(X ≤ x) = p.
///
/// # Panics
/// Panics if:
/// - `p` is not between 0 and 1
/// - `lambda` is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::{exponential_inverse_cdf, exponential_cdf};
///
/// // Calculate inverse CDF for p = 0.5 with rate parameter lambda = 2.0
/// let x = exponential_inverse_cdf(0.5, 2.0);
/// 
/// // Verify that CDF(x) is approximately p
/// let p = exponential_cdf(x, 2.0);
/// assert!((p - 0.5).abs() < 1e-10);
/// ```
pub fn exponential_inverse_cdf(p: f64, lambda: f64) -> f64 {
    assert!(p >= 0.0 && p <= 1.0, "p must be between 0 and 1");
    assert!(lambda > 0.0, "lambda must be positive");
    
    -((1.0 - p).ln()) / lambda
}

/// Mean (expected value) of the Exponential distribution.
///
/// Calculates the mean of an exponential distribution with rate parameter `lambda`.
///
/// # Arguments
/// * `lambda` - The rate parameter (must be positive)
///
/// # Returns
/// The mean of the distribution.
///
/// # Panics
/// Panics if `lambda` is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::exponential_mean;
///
/// // Mean of exponential distribution with rate parameter lambda = 2.0
/// let mean = exponential_mean(2.0);
/// assert!((mean - 0.5).abs() < 1e-10);
/// ```
pub fn exponential_mean(lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    
    1.0 / lambda
}

/// Variance of the Exponential distribution.
///
/// Calculates the variance of an exponential distribution with rate parameter `lambda`.
///
/// # Arguments
/// * `lambda` - The rate parameter (must be positive)
///
/// # Returns
/// The variance of the distribution.
///
/// # Panics
/// Panics if `lambda` is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::exponential_variance;
///
/// // Variance of exponential distribution with rate parameter lambda = 2.0
/// let variance = exponential_variance(2.0);
/// assert!((variance - 0.25).abs() < 1e-10);
/// ```
pub fn exponential_variance(lambda: f64) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    
    1.0 / (lambda * lambda)
}

/// Generate a random sample from an Exponential distribution.
///
/// # Arguments
/// * `lambda` - The rate parameter (must be positive)
/// * `rng` - A random number generator
///
/// # Returns
/// A random value from the exponential distribution.
///
/// # Panics
/// Panics if `lambda` is not positive
///
/// # Examples
/// ```
/// use rs_stats::distributions::exponential_distribution::exponential_sample;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let sample = exponential_sample(2.0, &mut rng);
/// assert!(sample >= 0.0); // Exponential distribution is always non-negative
/// ```
pub fn exponential_sample<R: Rng + ?Sized>(lambda: f64, rng: &mut R) -> f64 {
    assert!(lambda > 0.0, "lambda must be positive");
    
    let exp = Exp::new(lambda).unwrap();
    exp.sample(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    const EPSILON: f64 = 1e-10;
    
    #[test]
    fn test_exponential_pdf() {
        let lambda = 2.0;
        
        // PDF at x = 0
        let result = exponential_pdf(0.0, lambda);
        assert_eq!(result, lambda);
        
        // PDF at x = 1
        let result = exponential_pdf(1.0, lambda);
        let expected = lambda * (-lambda).exp();
        assert!((result - expected).abs() < EPSILON);
        
        // PDF at x = 0.5
        let result = exponential_pdf(0.5, lambda);
        let expected = lambda * (-lambda * 0.5).exp();
        assert!((result - expected).abs() < EPSILON);
    }
    
    #[test]
    fn test_exponential_cdf() {
        let lambda = 2.0;
        
        // CDF at x = 0
        let result = exponential_cdf(0.0, lambda);
        assert!((result - 0.0).abs() < EPSILON);
        
        // CDF at x = 1
        let result = exponential_cdf(1.0, lambda);
        let expected = 1.0 - (-lambda).exp();
        assert!((result - expected).abs() < EPSILON);
        
        // CDF at x = 0.5
        let result = exponential_cdf(0.5, lambda);
        let expected = 1.0 - (-lambda * 0.5).exp();
        assert!((result - expected).abs() < EPSILON);
    }
    
    #[test]
    fn test_exponential_inverse_cdf() {
        let lambda = 2.0;
        
        // Test inverse CDF with various probabilities
        let test_cases = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        
        for p in test_cases {
            let x = exponential_inverse_cdf(p, lambda);
            let cdf = exponential_cdf(x, lambda);
            assert!((cdf - p).abs() < EPSILON, 
                    "Inverse CDF failed for p = {}: got {}, expected {}", p, cdf, p);
        }
    }
    
    #[test]
    fn test_exponential_mean() {
        let lambda = 2.0;
        let result = exponential_mean(lambda);
        let expected = 1.0 / lambda;
        assert!((result - expected).abs() < EPSILON);
    }
    
    #[test]
    fn test_exponential_variance() {
        let lambda = 2.0;
        let result = exponential_variance(lambda);
        let expected = 1.0 / (lambda * lambda);
        assert!((result - expected).abs() < EPSILON);
    }
    
    #[test]
    fn test_exponential_sample() {
        let lambda = 2.0;
        let mut rng = thread_rng();
        
        // Test that samples are non-negative
        for _ in 0..100 {
            let sample = exponential_sample(lambda, &mut rng);
            assert!(sample >= 0.0);
        }
        
        // Test that mean of samples is close to theoretical mean
        // This is a statistical test, so we allow some margin of error
        let num_samples = 10000;
        let mut sum = 0.0;
        
        for _ in 0..num_samples {
            sum += exponential_sample(lambda, &mut rng);
        }
        
        let sample_mean = sum / (num_samples as f64);
        let theoretical_mean = exponential_mean(lambda);
        
        // Allow a 10% margin of error for the mean
        assert!((sample_mean - theoretical_mean).abs() < theoretical_mean * 0.1,
                "Sample mean {} is too far from theoretical mean {}", 
                sample_mean, theoretical_mean);
    }
    
    #[test]
    #[should_panic(expected = "lambda must be positive")]
    fn test_exponential_pdf_invalid_lambda() {
        exponential_pdf(1.0, -2.0);
    }
    
    #[test]
    #[should_panic(expected = "x must be non-negative")]
    fn test_exponential_pdf_invalid_x() {
        exponential_pdf(-1.0, 2.0);
    }
    
    #[test]
    fn test_exponential_config() {
        // Valid config
        let config = ExponentialConfig::new(2.0);
        assert!(config.is_some());
        
        // Invalid config
        let config = ExponentialConfig::new(0.0);
        assert!(config.is_none());
        
        let config = ExponentialConfig::new(-1.0);
        assert!(config.is_none());
    }
}

