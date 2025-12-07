//! # Cumulative Distribution Function (CDF)
//!
//! This module implements the cumulative distribution function for normal distributions.
//!
//! ## Mathematical Definition
//! The CDF of a normal distribution is defined as:
//!
//! Φ(x) = 0.5 * (1 + erf((x - μ) / (σ√2)))
//!
//! where:
//! - x is the value at which to evaluate the CDF
//! - μ is the mean of the distribution
//! - σ is the standard deviation of the distribution
//! - erf is the error function
//!
//! ## Key Properties
//! - Φ(-∞) = 0
//! - Φ(∞) = 1
//! - Φ(μ) = 0.5
//! - Φ(μ + a) = 1 - Φ(μ - a) (symmetry property)

use num_traits::ToPrimitive;
use crate::prob::erf;
use crate::utils::constants::SQRT_2;
use crate::error::{StatsResult, StatsError};

/// Calculate the cumulative distribution function (CDF) for a normal distribution
///
/// The CDF gives the probability that a random variable takes a value less than or equal to x.
///
/// # Arguments
/// * `x` - The value at which to evaluate the CDF
/// * `avg` - The mean (μ) of the distribution
/// * `stddev` - The standard deviation (σ) of the distribution
///
/// # Returns
/// The probability that a random variable is less than or equal to x
///
/// # Examples
/// ```
/// use rs_stats::prob::cumulative_distrib;
///
/// // Calculate CDF for standard normal distribution at x = 0
/// let cdf = cumulative_distrib(0.0, 0.0, 1.0).unwrap();
/// assert!((cdf - 0.5).abs() < 1e-8);
///
/// // Calculate CDF for non-standard normal distribution
/// let cdf = cumulative_distrib(12.0, 10.0, 2.0).unwrap();
/// assert!((cdf - 0.841344746).abs() < 1e-8);
/// ```
#[inline]
pub fn cumulative_distrib<T>(x: T, avg: f64, stddev: f64) -> StatsResult<f64> where T: ToPrimitive {
    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "prob::cumulative_distrib: Failed to convert x to f64".to_string(),
    })?;

    if stddev == 0.0 {
        return Err(StatsError::InvalidInput {
            message: "prob::cumulative_distrib: Standard deviation must be non-zero".to_string(),
        });
    }

    // Inline z-score calculation and combine with SQRT_2 division
    let z = (x_64 - avg) / (stddev * SQRT_2);
    Ok((1.0 + erf(z)?) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-5;

    #[test]
    fn test_cdf_standard_normal_at_mean() {
        // For a standard normal distribution (avg = 0, stddev = 1)
        // The CDF at the mean (0) should be 0.5
        let x = 0.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cumulative_distrib(x, avg, stddev).unwrap();
        let expected = 0.5;
        assert!(
            (result - expected).abs() < EPSILON,
            "CDF at the mean should be 0.5"
        );
    }

    #[test]
    fn test_cdf_standard_normal_positive() {
        // For a standard normal distribution (avg = 0, stddev = 1)
        // The CDF at 1.0 (z = 1.0) is approximately 0.841344746
        let x = 1.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cumulative_distrib(x, avg, stddev).unwrap();
        let expected = 0.841344746;
        assert!(
            (result - expected).abs() < EPSILON,
            "CDF for z = 1.0 should match expected"
        );
    }

    #[test]
    fn test_cdf_standard_normal_negative() {
        // For a standard normal distribution (avg = 0, stddev = 1)
        // The CDF at -1.0 (z = -1.0) is approximately 0.158655254
        let x = -1.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cumulative_distrib(x, avg, stddev).unwrap();
        let expected = 0.158655254;
        assert!(
            (result - expected).abs() < EPSILON,
            "CDF for z = -1.0 should match expected"
        );
    }

    #[test]
    fn test_cdf_non_standard_distribution() {
        // For a normal distribution with avg = 10, stddev = 2
        // We can compute the CDF for x = 12, which should give the same result as z = 1.0 for a standard normal distribution
        let x = 12.0;
        let avg = 10.0;
        let stddev = 2.0;
        let result = cumulative_distrib(x, avg, stddev).unwrap();
        let expected = 0.841344746; // CDF for z = 1.0 in standard normal
        assert!(
            (result - expected).abs() < EPSILON,
            "CDF for x = 12 in normal distribution with mean 10 and stddev 2 should match expected"
        );
    }

    #[test]
    fn test_cdf_large_positive_x() {
        // For a normal distribution (avg = 0, stddev = 1), a very large positive x should have a CDF close to 1
        let x = 5.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cumulative_distrib(x, avg, stddev).unwrap();
        let expected = 0.999999713; // Approximate value of CDF(5.0)
        assert!(
            (result - expected).abs() < EPSILON,
            "CDF for x = 5.0 should be very close to 1"
        );
    }

    #[test]
    fn test_cdf_large_negative_x() {
        // For a normal distribution (avg = 0, stddev = 1), a very large negative x should have a CDF close to 0
        let x = -5.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cumulative_distrib(x, avg, stddev).unwrap();
        println!("resuilit large negatif : {:?}", result);
        let expected = 0.000000287; // Approximate value of CDF(-5.0)
        assert!(
            (result - expected).abs() < EPSILON,
            "CDF for x = -5.0 should be very close to 0"
        );
    }
}
