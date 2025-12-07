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

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use crate::error::{StatsResult, StatsError};

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
pub struct ExponentialConfig<T> where T: ToPrimitive {
    /// The rate parameter.
    pub lambda: T,
}

impl<T> ExponentialConfig<T> where T: ToPrimitive {
    /// Creates a new ExponentialConfig with validation
    ///
    /// # Arguments
    /// * `lambda` - The rate parameter (must be positive)
    ///
    /// # Returns
    /// `Some(ExponentialConfig)` if parameter is valid, `None` otherwise
    pub fn new(lambda: T) -> StatsResult<Self> {
        let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
            message: "ExponentialConfig::new: Failed to convert lambda to f64".to_string(),
        })?;

        if lambda_64 > 0.0 {
            Ok(Self { lambda })
        } else {
            Err(StatsError::InvalidInput {
                message: "ExponentialConfig::new: lambda must be positive".to_string(),
            })
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
/// let pdf = exponential_pdf(1.0, 2.0).unwrap();
/// assert!((pdf - 0.27067).abs() < 1e-5);
/// ```
pub fn exponential_pdf<T>(x: T, lambda: T) -> StatsResult<f64> where T: ToPrimitive {
    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_pdf: Failed to convert x to f64".to_string(),
    })?;

    if x_64 < 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_pdf: x must be non-negative".to_string(),
        });
    }

    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_pdf: Failed to convert lambda to f64".to_string(),
    })?;

    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_pdf: lambda must be positive".to_string(),
        });
    }

    Ok(if x_64 == 0.0 { lambda_64 } else { lambda_64 * (-lambda_64 * x_64).exp() })
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
/// let cdf = exponential_cdf(1.0, 2.0).unwrap();
/// assert!((cdf - 0.86466).abs() < 1e-5);
/// ```
pub fn exponential_cdf<T>(x: T, lambda: T) -> StatsResult<f64> where T: ToPrimitive {
    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_cdf: Failed to convert x to f64".to_string(),
    })?;

    if x_64 < 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_cdf: x must be non-negative".to_string(),
        });
    } 
    
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_cdf: Failed to convert lambda to f64".to_string(),
    })?;

    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_cdf: lambda must be positive".to_string(),
        });
    }
    Ok(1.0 - (-lambda_64 * x_64).exp())
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
/// let x = exponential_inverse_cdf(0.5, 2.0).unwrap();
///
/// // Verify that CDF(x) is approximately p
/// let p = exponential_cdf(x, 2.0).unwrap();
/// assert!((p - 0.5).abs() < 1e-10);
/// ```
pub fn exponential_inverse_cdf<T>(p: T, lambda: T) -> StatsResult<f64> where T: ToPrimitive {
    let p_64 = p.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_inverse_cdf: Failed to convert p to f64".to_string(),
    })?;

    if p_64 < 0.0 || p_64 > 1.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_inverse_cdf: p must be between 0 and 1".to_string(),
        });
    }
    
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_inverse_cdf: Failed to convert lambda to f64".to_string(),
    })?;

    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_inverse_cdf: lambda must be positive".to_string(),
        });
    }
    Ok(-((1.0 - p_64).ln()) / lambda_64)
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
/// let mean = exponential_mean(2.0).unwrap();
/// assert!((mean - 0.5).abs() < 1e-10);
/// ```
pub fn exponential_mean<T>(lambda: T) -> StatsResult<f64> where T: ToPrimitive {
    let lambda_64 = lambda.to_f64().ok_or_else(|| StatsError::ConversionError{
        message: "exponential_mean: Failed to convert lambda to f64".to_string(),
    })?;
    if lambda_64 <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_mean: lambda must be positive".to_string(),
        });
    }

    Ok(1.0 / lambda_64)
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
/// let variance = exponential_variance(2.0).unwrap();
/// assert!((variance - 0.25).abs() < 1e-10);
/// ```
pub fn exponential_variance(lambda: f64) -> StatsResult<f64> {
    if lambda <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "exponential_variance: lambda must be positive".to_string(),
        });
    }

    Ok(1.0 / (lambda * lambda)) 
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_exponential_pdf() {
        let lambda = 2.0;

        // PDF at x = 0
        let result = exponential_pdf(0.0, lambda).unwrap();
        assert_eq!(result, lambda);

        // PDF at x = 1
        let result = exponential_pdf(1.0, lambda).unwrap();
        let expected = lambda * (-lambda).exp();
        assert!((result - expected).abs() < EPSILON);

        // PDF at x = 0.5
        let result = exponential_pdf(0.5, lambda).unwrap();
        let expected = lambda * (-lambda * 0.5).exp();
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_exponential_cdf() {
        let lambda = 2.0_f64;

        // CDF at x = 0
        let result = exponential_cdf(0.0, lambda).unwrap();
        assert!((result - 0.0).abs() < EPSILON);

        // CDF at x = 1
        let result = exponential_cdf(1.0, lambda).unwrap();
        let expected = 1.0 - (-lambda).exp();
        assert!((result - expected).abs() < EPSILON);

        // CDF at x = 0.5
        let result = exponential_cdf(0.5, lambda).unwrap();
        let expected = 1.0 - (-lambda * 0.5).exp();
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_exponential_inverse_cdf() {
        let lambda = 2.0_f64;

        // Test inverse CDF with various probabilities
        let test_cases = vec![0.1, 0.25, 0.5, 0.75, 0.9];

        for p in test_cases {
            let x = exponential_inverse_cdf(p, lambda).unwrap();
            let cdf = exponential_cdf(x, lambda).unwrap();
            assert!(
                (cdf - p).abs() < EPSILON,
                "Inverse CDF failed for p = {}: got {}, expected {}",
                p,
                cdf,
                p
            );
        }
    }

    #[test]
    fn test_exponential_mean() {
        let lambda = 2.0;
        let result = exponential_mean(lambda).unwrap();
        let expected = 1.0 / lambda;
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_exponential_variance() {
        let lambda = 2.0;
        let result = exponential_variance(lambda).unwrap();
        let expected = 1.0 / (lambda * lambda);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_exponential_pdf_invalid_lambda() {
        let result = exponential_pdf(1.0, -2.0);
        assert!(result.is_err());
        match result {
            Err(StatsError::InvalidInput { message }) => {
                assert!(message.contains("lambda must be positive"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_exponential_pdf_invalid_x() {
        let result = exponential_pdf(-1.0, 2.0);
        assert!(result.is_err());
        match result {
            Err(StatsError::InvalidInput { message }) => {
                assert!(message.contains("x must be non-negative"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_exponential_config() {
        // Valid config
        let config = ExponentialConfig::new(2.0);
        assert!(config.is_ok());

        // Invalid config
        let config = ExponentialConfig::new(0.0);
        assert!(config.is_err());

        let config = ExponentialConfig::new(-1.0);
        assert!(config.is_err());
    }
}
