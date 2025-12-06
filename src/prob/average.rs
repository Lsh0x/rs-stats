//! # Average Calculation
//!
//! This module provides functions for calculating the arithmetic mean (average) of a dataset.
//!
//! The average is calculated as the sum of all values divided by the number of values.
//!
//! ## Supported Types
//! The average function accepts any numeric type that implements `num_traits::ToPrimitive`,
//! including:
//! - Primitive integers (i8, i16, i32, i64, u8, u16, u32, u64)
//! - Floating point numbers (f32, f64)
//! - Big integers (BigInt, BigUint)
//! - Any custom type that implements ToPrimitive

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;
use std::fmt::Debug;

/// Calculate the arithmetic mean (average) of a dataset.
///
/// The average is calculated as the sum of all values divided by the number of values.
///
/// # Arguments
/// * `data` - A slice of numeric values implementing `ToPrimitive`
///
/// # Returns
/// * `StatsResult<f64>` - The average as a `f64`, or an error if the input is invalid
///
/// # Errors
/// Returns `StatsError::EmptyData` if the input slice is empty.
/// Returns `StatsError::ConversionError` if any value cannot be converted to f64.
///
/// # Examples
/// ```
/// use rs_stats::prob::average;
///
/// // Calculate average of integers
/// let int_data = [1, 2, 3, 4, 5];
/// let avg = average(&int_data)?;
/// println!("Average of integers: {}", avg);
///
/// // Calculate average of floats
/// let float_data = [1.0, 2.5, 3.0, 4.5, 5.0];
/// let avg = average(&float_data)?;
/// println!("Average of floats: {}", avg);
///
/// // Handle empty input
/// let empty_data: &[i32] = &[];
/// assert!(average(empty_data).is_err());
/// # Ok::<(), rs_stats::StatsError>(())
/// ```
#[inline]
pub fn average<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    if data.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot calculate average of empty dataset"
        ));
    }
    let sum: f64 = data
        .iter()
        .enumerate()
        .map(|(i, x)| {
            x.to_f64().ok_or_else(|| StatsError::conversion_error(format!(
                "Failed to convert value at index {} to f64",
                i
            )))
        })
        .collect::<Result<Vec<f64>, _>>()?
        .iter()
        .sum();
    Ok(sum / data.len() as f64)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_integers() {
        let data = vec![1, 2, 3, 4, 5];
        let result = average(&data).unwrap();
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_average_floats() {
        let data = vec![1.5, 2.5, 3.5, 4.5];
        let result = average(&data).unwrap();
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_average_mixed_types() {
        let data = vec![1.0, 2.0, 3.0, 4.5, 5.5]; // All elements are f64
        let result = average(&data).unwrap();
        assert_eq!(result, 3.2);
    }

    #[test]
    fn test_average_empty_slice() {
        let data: Vec<f64> = Vec::new();
        let result = average(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::EmptyData { .. }
        ));
    }

    #[test]
    fn test_average_single_value() {
        let data = vec![10.0];
        let result = average(&data).unwrap();
        assert_eq!(result, 10.0);
    }
}
