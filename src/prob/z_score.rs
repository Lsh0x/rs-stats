//! # Z-Score Calculation
//!
//! This module implements the z-score (standard score) calculation,
//! which measures how many standard deviations a value is from the mean.
//!
//! ## Mathematical Definition
//! The z-score is defined as:
//!
//! z = (x - μ) / σ
//!
//! where:
//! - x is the raw score
//! - μ is the population mean
//! - σ is the population standard deviation
//!
//! ## Key Properties
//! - z-scores have a mean of 0 and standard deviation of 1
//! - Positive z-scores indicate values above the mean
//! - Negative z-scores indicate values below the mean
//! - z-scores are unitless and allow comparison across different distributions

use num_traits::ToPrimitive;
use crate::error::{StatsResult, StatsError};

/// Calculate the z-score (standard score) of a value
///
/// The z-score indicates how many standard deviations a value is from the mean.
///
/// # Arguments
/// * `x` - The value to standardize
/// * `avg` - The mean (μ) of the distribution
/// * `stddev` - The standard deviation (σ) of the distribution
///
/// # Returns
/// The z-score of the value. Returns infinity if stddev is 0.
///
/// # Examples
/// ```
/// use rs_stats::prob::z_score;
///
/// // Calculate z-score for a value above the mean
/// let z = z_score(85.0, 70.0, 10.0).unwrap();
/// assert!((z - 1.5).abs() < 1e-10);
///
/// // Calculate z-score for a value below the mean
/// let z = z_score(55.0, 70.0, 10.0).unwrap();
/// assert!((z - (-1.5)).abs() < 1e-10);
///
/// // Handle zero standard deviation case
/// let z = z_score(70.0, 70.0, 0.0).unwrap();
/// assert!(z.is_infinite());
/// ```
#[inline]
pub fn z_score<T>(x: T, avg: f64, stddev: f64) -> StatsResult<f64> where T: ToPrimitive {

    if stddev == 0.0 {
        return Ok(f64::INFINITY);
    }

    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "prob::z_score: Failed to convert x to f64".to_string(),
    })?;

    Ok((x_64 - avg) / stddev)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_z_score_integer() {
        let x = 5.0;
        let avg = 3.0;
        let stddev = 2.0;
        let result = z_score(x, avg, stddev).unwrap();
        let expected = (5.0 - 3.0) / 2.0; // (x - avg) / stddev
        assert!(
            (result - expected).abs() < EPSILON,
            "Z-score for value 5 with avg 3 and stddev 2 should match expected"
        );
    }

    #[test]
    fn test_z_score_float() {
        let x = 4.5;
        let avg = 3.0;
        let stddev = 1.5;
        let result = z_score(x, avg, stddev).unwrap();
        let expected = (4.5 - 3.0) / 1.5; // (x - avg) / stddev
        assert!(
            (result - expected).abs() < EPSILON,
            "Z-score for value 4.5 with avg 3 and stddev 1.5 should match expected"
        );
    }

    #[test]
    fn test_z_score_negative() {
        let x = 1.0;
        let avg = 3.0;
        let stddev = 2.0;
        let result = z_score(x, avg, stddev).unwrap();
        let expected = (1.0 - 3.0) / 2.0; // (x - avg) / stddev
        assert!(
            (result - expected).abs() < EPSILON,
            "Z-score for value 1 with avg 3 and stddev 2 should match expected"
        );
    }

    #[test]
    fn test_z_score_zero_stddev() {
        let x = 3.0;
        let avg = 3.0;
        let stddev = 0.0;
        let result = z_score(x, avg, stddev).unwrap();
        assert!(
            result.is_infinite(),
            "Z-score should be infinite when stddev is 0"
        );
    }

    #[test]
    fn test_z_score_zero_mean() {
        let x = 3.0;
        let avg = 0.0;
        let stddev = 2.0;
        let result = z_score(x, avg, stddev).unwrap();
        let expected = (3.0 - 0.0) / 2.0;
        assert!(
            (result - expected).abs() < EPSILON,
            "Z-score for value 3 with avg 0 and stddev 2 should match expected"
        );
    }
}
