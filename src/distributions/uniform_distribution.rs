//! # Uniform Distribution
//!
//! This module implements the Uniform distribution, a continuous probability distribution
//! where all values within a given range are equally likely to occur.
//!
//! ## Key Characteristics
//! - All values in the range [a, b] have equal probability
//! - Constant PDF within the range, zero outside
//! - Linear CDF function from 0 to 1 over the range
//!
//! ## Common Applications
//! - Modeling random variables with no bias toward any value in a range
//! - Generating random numbers for Monte Carlo simulations
//! - Baseline distribution for comparing other distributions
//!
//! ## Mathematical Formulation
//! For a uniform distribution over the interval [a, b]:
//!
//! PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
//! CDF: F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
//! Mean: (a + b)/2
//! Variance: (b - a)²/12

use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

/// Configuration for the Uniform distribution.
///
/// # Fields
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be greater than a)
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::UniformConfig;
///
/// let config = UniformConfig { a: 0.0, b: 1.0 };
/// assert!(config.a < config.b);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UniformConfig {
    /// The lower bound of the distribution.
    pub a: f64,
    /// The upper bound of the distribution.
    pub b: f64,
}

impl UniformConfig {
    /// Creates a new UniformConfig with validation
    ///
    /// # Arguments
    /// * `a` - The lower bound of the distribution
    /// * `b` - The upper bound of the distribution
    ///
    /// # Returns
    /// `Some(UniformConfig)` if parameters are valid (a < b), `None` otherwise
    pub fn new(a: f64, b: f64) -> Option<Self> {
        if a < b { Some(Self { a, b }) } else { None }
    }
}

/// Probability density function (PDF) for the Uniform distribution.
///
/// Calculates the probability density at value `x` for a uniform distribution
/// over the interval [a, b].
///
/// # Arguments
/// * `x` - The value at which to evaluate the PDF
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be > a)
///
/// # Returns
/// The probability density at point x:
/// * 1/(b-a) if a ≤ x ≤ b
/// * 0 otherwise
///
/// # Panics
/// Panics if a ≥ b
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::uniform_pdf;
///
/// // Calculate PDF for x = 0.5 in uniform distribution from 0 to 1
/// let pdf = uniform_pdf(0.5, 0.0, 1.0);
/// assert!((pdf - 1.0).abs() < 1e-10);
///
/// // PDF is 0 outside the range
/// let pdf = uniform_pdf(1.5, 0.0, 1.0);
/// assert!((pdf - 0.0).abs() < 1e-10);
/// ```
pub fn uniform_pdf(x: f64, a: f64, b: f64) -> f64 {
    assert!(a < b, "Lower bound must be less than upper bound");

    if x < a || x > b { 0.0 } else { 1.0 / (b - a) }
}

/// Cumulative distribution function (CDF) for the Uniform distribution.
///
/// Calculates the probability that a random variable from the uniform
/// distribution on [a, b] is less than or equal to `x`.
///
/// # Arguments
/// * `x` - The value at which to evaluate the CDF
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be > a)
///
/// # Returns
/// The cumulative probability F(x):
/// * 0 if x < a
/// * (x-a)/(b-a) if a ≤ x ≤ b
/// * 1 if x > b
///
/// # Panics
/// Panics if a ≥ b
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::uniform_cdf;
///
/// // Calculate CDF for x = 0.5 in uniform distribution from 0 to 1
/// let cdf = uniform_cdf(0.5, 0.0, 1.0);
/// assert!((cdf - 0.5).abs() < 1e-10);
///
/// // CDF is 0 below the range
/// let cdf = uniform_cdf(-0.5, 0.0, 1.0);
/// assert!((cdf - 0.0).abs() < 1e-10);
///
/// // CDF is 1 above the range
/// let cdf = uniform_cdf(1.5, 0.0, 1.0);
/// assert!((cdf - 1.0).abs() < 1e-10);
/// ```
pub fn uniform_cdf(x: f64, a: f64, b: f64) -> f64 {
    assert!(a < b, "Lower bound must be less than upper bound");

    if x < a {
        0.0
    } else if x > b {
        1.0
    } else {
        (x - a) / (b - a)
    }
}

/// Inverse cumulative distribution function (Quantile Function) for the Uniform distribution.
///
/// Calculates the value `x` such that P(X ≤ x) = p for a uniform
/// random variable X on [a, b].
///
/// # Arguments
/// * `p` - The probability (must be between 0 and 1)
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be > a)
///
/// # Returns
/// The value x such that P(X ≤ x) = p
///
/// # Panics
/// Panics if:
/// - a ≥ b
/// - p < 0 or p > 1
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::{uniform_cdf, uniform_inverse_cdf};
///
/// // Calculate the value at which CDF = 0.7 in uniform distribution from 0 to 1
/// let x = uniform_inverse_cdf(0.7, 0.0, 1.0);
/// assert!((x - 0.7).abs() < 1e-10);
///
/// // Verify inverse relationship with CDF
/// let p = 0.3;
/// let a = 2.0;
/// let b = 5.0;
/// let x = uniform_inverse_cdf(p, a, b);
/// assert!((uniform_cdf(x, a, b) - p).abs() < 1e-10);
/// ```
pub fn uniform_inverse_cdf(p: f64, a: f64, b: f64) -> f64 {
    assert!(a < b, "Lower bound must be less than upper bound");
    assert!(
        (0.0..=1.0).contains(&p),
        "Probability must be between 0 and 1"
    );

    a + (p * (b - a))
}

/// Calculate the mean of a Uniform distribution.
///
/// # Arguments
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be > a)
///
/// # Returns
/// The mean of the uniform distribution: (a + b) / 2
///
/// # Panics
/// Panics if a ≥ b
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::uniform_mean;
///
/// let mean = uniform_mean(0.0, 1.0);
/// assert!((mean - 0.5).abs() < 1e-10);
///
/// let mean = uniform_mean(-3.0, 3.0);
/// assert!((mean - 0.0).abs() < 1e-10);
/// ```
pub fn uniform_mean(a: f64, b: f64) -> f64 {
    assert!(a < b, "Lower bound must be less than upper bound");

    (a + b) / 2.0
}

/// Calculate the variance of a Uniform distribution.
///
/// # Arguments
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be > a)
///
/// # Returns
/// The variance of the uniform distribution: (b - a)² / 12
///
/// # Panics
/// Panics if a ≥ b
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::uniform_variance;
///
/// let variance = uniform_variance(0.0, 1.0);
/// assert!((variance - 1.0/12.0).abs() < 1e-10);
///
/// let variance = uniform_variance(0.0, 12.0);
/// assert!((variance - 12.0).abs() < 1e-10);
/// ```
pub fn uniform_variance(a: f64, b: f64) -> f64 {
    assert!(a < b, "Lower bound must be less than upper bound");

    ((b - a) * (b - a)) / 12.0
}

/// Generate a random sample from a Uniform distribution.
///
/// # Arguments
/// * `a` - The lower bound of the distribution
/// * `b` - The upper bound of the distribution (must be > a)
/// * `rng` - Random number generator
///
/// # Returns
/// A random value from the uniform distribution on [a, b]
///
/// # Panics
/// Panics if a ≥ b
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::uniform_sample;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let sample = uniform_sample(0.0, 1.0, &mut rng);
/// assert!(sample >= 0.0 && sample <= 1.0);
/// ```
pub fn uniform_sample<R: Rng + ?Sized>(a: f64, b: f64, rng: &mut R) -> f64 {
    assert!(a < b, "Lower bound must be less than upper bound");

    let dist = Uniform::new(a, b);
    dist.sample(rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_uniform_pdf_inside_range() {
        // For a uniform distribution on [0, 1], PDF should be 1 inside the range
        assert!((uniform_pdf(0.0, 0.0, 1.0) - 1.0).abs() < EPSILON);
        assert!((uniform_pdf(0.5, 0.0, 1.0) - 1.0).abs() < EPSILON);
        assert!((uniform_pdf(1.0, 0.0, 1.0) - 1.0).abs() < EPSILON);

        // For a uniform distribution on [2, 4], PDF should be 1/2 inside the range
        assert!((uniform_pdf(2.0, 2.0, 4.0) - 0.5).abs() < EPSILON);
        assert!((uniform_pdf(3.0, 2.0, 4.0) - 0.5).abs() < EPSILON);
        assert!((uniform_pdf(4.0, 2.0, 4.0) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_pdf_outside_range() {
        // PDF should be 0 outside the range
        assert!((uniform_pdf(-1.0, 0.0, 1.0) - 0.0).abs() < EPSILON);
        assert!((uniform_pdf(2.0, 0.0, 1.0) - 0.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_uniform_pdf_invalid_range() {
        // Should panic if a >= b
        uniform_pdf(0.5, 1.0, 0.0);
    }

    #[test]
    fn test_uniform_cdf_inside_range() {
        // For a uniform distribution on [0, 1], CDF should increase linearly from 0 to 1
        assert!((uniform_cdf(0.0, 0.0, 1.0) - 0.0).abs() < EPSILON);
        assert!((uniform_cdf(0.25, 0.0, 1.0) - 0.25).abs() < EPSILON);
        assert!((uniform_cdf(0.5, 0.0, 1.0) - 0.5).abs() < EPSILON);
        assert!((uniform_cdf(0.75, 0.0, 1.0) - 0.75).abs() < EPSILON);
        assert!((uniform_cdf(1.0, 0.0, 1.0) - 1.0).abs() < EPSILON);

        // For a uniform distribution on [2, 4]
        assert!((uniform_cdf(2.0, 2.0, 4.0) - 0.0).abs() < EPSILON);
        assert!((uniform_cdf(3.0, 2.0, 4.0) - 0.5).abs() < EPSILON);
        assert!((uniform_cdf(4.0, 2.0, 4.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_cdf_outside_range() {
        // CDF should be 0 below the range
        assert!((uniform_cdf(-1.0, 0.0, 1.0) - 0.0).abs() < EPSILON);

        // CDF should be 1 above the range
        assert!((uniform_cdf(2.0, 0.0, 1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_cdf_inverse_cdf_relationship() {
        // Test that CDF and inverse CDF are inverses of each other
        let a = 2.0;
        let b = 5.0;

        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let x = uniform_inverse_cdf(p, a, b);
            let p_result = uniform_cdf(x, a, b);
            assert!(
                (p - p_result).abs() < EPSILON,
                "CDF(inverse_CDF(p)) should equal p"
            );

            // Also verify that inverse_CDF(CDF(x)) ≈ x for points within the range
            if p > 0.0 && p < 1.0 {
                let x_within_range = a + p * (b - a);
                let p_cdf = uniform_cdf(x_within_range, a, b);
                let x_result = uniform_inverse_cdf(p_cdf, a, b);
                assert!(
                    (x_within_range - x_result).abs() < EPSILON,
                    "inverse_CDF(CDF(x)) should equal x"
                );
            }
        }
    }
}
