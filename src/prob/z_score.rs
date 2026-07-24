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

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;

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
/// The z-score of the value.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `stddev` is zero, negative, or not
/// finite: a z-score is undefined without positive dispersion (the old
/// behaviour silently returned `+∞`, even for the 0/0 case `x == avg`, and
/// accepted a negative σ that flipped the sign of every score).
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
/// // Zero standard deviation is an error, not infinity
/// assert!(z_score(70.0, 70.0, 0.0).is_err());
/// ```
#[inline]
pub fn z_score<T>(x: T, avg: f64, stddev: f64) -> StatsResult<f64>
where
    T: ToPrimitive,
{
    if stddev <= 0.0 || !stddev.is_finite() {
        return Err(StatsError::invalid_input(format!(
            "prob::z_score: standard deviation must be positive and finite, got {stddev}"
        )));
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
    fn test_z_score_invalid_stddev() {
        // σ = 0 (including the 0/0 case x == avg), σ < 0 and non-finite σ
        // are all undefined — typed errors, not ±∞.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                z_score(3.0, 3.0, bad).is_err(),
                "z_score should reject stddev = {bad}"
            );
        }
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
