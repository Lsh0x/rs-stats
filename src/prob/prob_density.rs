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

use crate::error::{StatsError, StatsResult};
use crate::utils::constants::INV_SQRT_2PI;
use num_traits::ToPrimitive;

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
/// let pdf = probability_density(0.0, 0.0, 1.0).unwrap();
/// assert!((pdf - 0.3989422804014327).abs() < 1e-10);
///
/// // Calculate PDF at x = 1 for N(0,1)
/// let pdf = probability_density(1.0, 0.0, 1.0).unwrap();
/// assert!((pdf - 0.24197072451914337).abs() < 1e-10);
/// ```
#[inline]
pub fn probability_density<T>(x: T, avg: f64, stddev: f64) -> StatsResult<f64>
where
    T: ToPrimitive,
{
    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "prob::probability_density: Failed to convert x to f64".to_string(),
    })?;

    if stddev == 0.0 {
        return Err(StatsError::InvalidInput {
            message: "prob::probability_density: Standard deviation must be non-zero".to_string(),
        });
    }
    // Inline z-score calculation instead of calling z_score()
    let z = (x_64 - avg) / stddev;
    // Use multiplication instead of powi(2) for better performance
    let exponent = -0.5 * z * z;

    Ok(exponent.exp() * INV_SQRT_2PI / stddev)
}

/// normal_probability_density return the PDF with z already normalized
/// <https://en.wikipedia.org/wiki/Probability_density_function>
#[inline]
pub fn normal_probability_density(z: f64) -> StatsResult<f64> {
    let exponent = -0.5 * z * z;
    Ok(exponent.exp() * INV_SQRT_2PI)
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
            let actual = probability_density(x, avg, stddev).unwrap();
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
        let actual = probability_density(x, avg, stddev).unwrap();
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
        let actual = probability_density(x, avg, stddev).unwrap();
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
            let actual = normal_probability_density(z).unwrap();
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
        let actual =
            normal_probability_density(z).unwrap() - normal_probability_density(-z).unwrap();
        assert!(
            actual.abs() < 1e-10,
            "normal_probability_density(z) should equal normal_probability_density(-z), but got {}",
            actual
        );
    }

    #[test]
    fn test_normal_probability_density_limits() {
        // Test approaching limits
        assert!(normal_probability_density(10.0).unwrap() < 1e-20); // PDF -> 0 as z -> +/-inf
        assert!(normal_probability_density(-10.0).unwrap() < 1e-20);
    }

    #[test]
    fn test_probability_density_stddev_zero() {
        let result = probability_density(0.0, 0.0, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }
}
