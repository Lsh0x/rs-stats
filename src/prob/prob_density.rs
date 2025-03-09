//! # Probability Density Functions
//!
//! This module provides implementations of probability density functions (PDFs) for normal distributions.
//!
//! The probability density function describes the relative likelihood for a random variable
//! to take on a given value. For a normal distribution, the PDF is given by:
//!
//! f(x|μ,σ) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))
//!
//! where:
//! - μ is the mean
//! - σ is the standard deviation
//! - x is the value at which to evaluate the PDF
//!
//! ## Functions
//! - `probability_density`: Calculates PDF for a given x, mean, and standard deviation
//! - `normal_probability_density`: Calculates PDF for a given z-score (pre-normalized value)

use crate::prob::z_score::z_score;
use std::f64::consts::PI;

/// Calculate the probability density function (PDF) for a normal distribution
///
/// # Arguments
/// * `x` - The value at which to evaluate the PDF
/// * `avg` - The mean (μ) of the distribution
/// * `stddev` - The standard deviation (σ) of the distribution
///
/// # Returns
/// The probability density at point x
///
/// # Examples
/// ```
/// use rs_stats::prob::probability_density;
///
/// // Calculate PDF at x = 0 for standard normal distribution
/// let pdf = probability_density(0.0, 0.0, 1.0);
/// assert!((pdf - 0.3989422804014327).abs() < 1e-10);
///
/// // Calculate PDF at x = 1 for N(0,1)
/// let pdf = probability_density(1.0, 0.0, 1.0);
/// assert!((pdf - 0.24197072451914337).abs() < 1e-10);
/// ```
#[inline]
pub fn probability_density(x: f64, avg: f64, stddev: f64) -> f64 {
    (z_score(x, avg, stddev).powi(2) / -2.0).exp() / (stddev * (PI * 2.0).sqrt())
}

/// normal_probability_density return the PDF with z already normalized
/// https://en.wikipedia.org/wiki/Probability_density_function
#[inline]
pub fn normal_probability_density(z: f64) -> f64 {
    (z.powi(2) / -2.0).exp() / (PI * 2.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_density_basic() {
        let avg = 0.0;
        let stddev = 1.0;

        let test_cases = vec![
            (0.0, 0.3989422804014327),   // Peak of the distribution
            (1.0, 0.24197072451914337),  // One standard deviation away
            (-1.0, 0.24197072451914337), // One standard deviation away (symmetry)
            (2.0, 0.05399096651318806),  // Two standard deviations away
            (3.0, 0.00443184841193801),  // Three standard deviations away
        ];

        for (x, expected) in test_cases {
            let actual = probability_density(x, avg, stddev);
            assert!(
                (actual - expected).abs() < 1e-10,
                "For x = {}, expected {}, but got {}",
                x,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_probability_density_different_mean() {
        let avg = 5.0;
        let stddev = 2.0;
        let x = 7.0; // One standard deviation above the mean
        let expected = 0.12098536225957168;
        let actual = probability_density(x, avg, stddev);
        assert!(
            (actual - expected).abs() < 1e-10,
            "For x = {}, expected {}, but got {}",
            x,
            expected,
            actual
        );
    }

    #[test]
    fn test_probability_density_different_stddev() {
        let avg = 0.0;
        let stddev = 0.5;
        let x = 0.0;
        let expected = 0.7978845608028654;
        let actual = probability_density(x, avg, stddev);
        assert!(
            (actual - expected).abs() < 1e-10,
            "For x = {}, expected {}, but got {}",
            x,
            expected,
            actual
        );
    }

    #[test]
    fn test_normal_probability_density_basic() {
        let test_cases = vec![
            (0.0, 0.3989422804014327),   // Peak of the distribution
            (1.0, 0.24197072451914337),  // One standard deviation away
            (-1.0, 0.24197072451914337), // One standard deviation away (symmetry)
            (2.0, 0.05399096651318806),  // Two standard deviations away
            (3.0, 0.00443184841193801),  // Three standard deviations away
        ];

        for (z, expected) in test_cases {
            let actual = normal_probability_density(z);
            assert!(
                (actual - expected).abs() < 1e-10,
                "For z = {}, expected {}, but got {}",
                z,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_normal_probability_density_symmetry() {
        let z = 0.7;
        let actual = normal_probability_density(z) - normal_probability_density(-z);
        assert!(
            actual.abs() < 1e-10,
            "normal_probability_density(z) should equal normal_probability_density(-z), but got {}",
            actual
        );
    }

    #[test]
    fn test_normal_probability_density_limits() {
        // Test approaching limits
        assert!(normal_probability_density(10.0) < 1e-20); // PDF -> 0 as z -> +/-inf
        assert!(normal_probability_density(-10.0) < 1e-20);
    }
}
