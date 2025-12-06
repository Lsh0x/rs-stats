//! # Variance Calculation
//!
//! This module provides functions for calculating the variance of a dataset.
//!
//! The variance is a measure of how spread out the numbers in a data set are.
//! It is calculated as the average of the squared differences from the mean.
//!
//! ## Supported Types
//! The variance function accepts any numeric type that implements `num_traits::ToPrimitive`,
//! including:
//! - Primitive integers (i8, i16, i32, i64, u8, u16, u32, u64)
//! - Floating point numbers (f32, f64)
//! - Big integers (BigInt, BigUint)
//! - Any custom type that implements ToPrimitive

use crate::error::{StatsError, StatsResult};
use crate::prob::average::average;
use num_traits::ToPrimitive;
use std::fmt::Debug;

/// Calculate the variance of a dataset.
///
/// The variance is a measure of how spread out the numbers in a data set are.
/// It is calculated as the average of the squared differences from the mean.
///
/// # Arguments
/// * `data` - A slice of numeric values implementing `ToPrimitive`
///
/// # Returns
/// * `StatsResult<f64>` - The variance as a `f64`, or an error if the input is invalid
///
/// # Errors
/// Returns `StatsError::EmptyData` if the input slice is empty.
/// Returns `StatsError::ConversionError` if any value cannot be converted to f64.
///
/// # Examples
/// ```
/// use rs_stats::prob::variance;
///
/// // Calculate variance of integers
/// let int_data = [1, 2, 3, 4, 5];
/// let var = variance(&int_data)?;
/// println!("Variance of integers: {}", var);
///
/// // Calculate variance of floats
/// let float_data = [1.0, 2.5, 3.0, 4.5, 5.0];
/// let var = variance(&float_data)?;
/// println!("Variance of floats: {}", var);
///
/// // Handle empty input
/// let empty_data: &[i32] = &[];
/// assert!(variance(empty_data).is_err());
/// # Ok::<(), rs_stats::StatsError>(())
/// ```
pub fn variance<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    if data.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot calculate variance of empty dataset"
        ));
    }

    let avg = average(data)?;
    let mut sum = 0.0;
    for (i, x) in data.iter().enumerate() {
        let x = x.to_f64().ok_or_else(|| StatsError::conversion_error(format!(
            "Failed to convert value at index {} to f64",
            i
        )))?;
        sum += (x - avg).powi(2);
    }
    Ok(sum / data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_integers() {
        let data = [1, 2, 3, 4, 5];
        let variance = variance(&data).unwrap();
        assert!(!variance.is_nan());
    }

    #[test]
    fn test_variance_floats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = variance(&data).unwrap();
        assert!(!variance.is_nan());
    }

    #[test]
    fn test_variance_mixed_floats() {
        let data = [1.0, 2.5, 3.0, 4.5, 5.0];
        let variance = variance(&data).unwrap();
        assert!(!variance.is_nan());
    }

    #[test]
    fn test_variance_single_value() {
        let data = [5];
        let variance = variance(&data).unwrap();
        assert_eq!(variance, 0.0);
    }

    #[test]
    fn test_variance_empty_slice() {
        let data: &[f64] = &[];
        let variance = variance(data);
        assert!(variance.is_err());
        assert!(matches!(
            variance.unwrap_err(),
            StatsError::EmptyData { .. }
        ));
    }

    #[test]
    fn test_variance_identical_values() {
        let data = [2.0; 10];
        let variance = variance(&data).unwrap();
        assert_eq!(variance, 0.0);
    }
}
