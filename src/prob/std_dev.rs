//! # Standard Deviation Calculation
//!
//! This module provides functions for calculating the standard deviation of a dataset.
//!
//! The standard deviation is a measure of the amount of variation or dispersion of a set of values.
//! It is calculated as the square root of the variance.
//!
//! ## Supported Types
//! The standard deviation function accepts any numeric type that implements `num_traits::ToPrimitive`,
//! including:
//! - Primitive integers (i8, i16, i32, i64, u8, u16, u32, u64)
//! - Floating point numbers (f32, f64)
//! - Big integers (BigInt, BigUint)
//! - Any custom type that implements ToPrimitive

use crate::error::StatsResult;
use crate::prob::variance::variance;
use num_traits::ToPrimitive;
use std::fmt::Debug;

/// Calculate the standard deviation of a dataset.
///
/// The standard deviation is a measure of the amount of variation or dispersion of a set of values.
/// It is calculated as the square root of the variance.
///
/// # Arguments
/// * `data` - A slice of numeric values implementing `ToPrimitive`
///
/// # Returns
/// * `StatsResult<f64>` - The standard deviation as a `f64`, or an error if the input is invalid
///
/// # Errors
/// Returns `StatsError::EmptyData` if the input slice is empty.
/// Returns `StatsError::ConversionError` if any value cannot be converted to f64.
///
/// # Examples
/// ```
/// use rs_stats::prob::std_dev;
///
/// // Calculate standard deviation of integers
/// let int_data = [1, 2, 3, 4, 5];
/// let sd = std_dev(&int_data)?;
/// println!("Standard deviation of integers: {}", sd);
///
/// // Calculate standard deviation of floats
/// let float_data = [1.0, 2.5, 3.0, 4.5, 5.0];
/// let sd = std_dev(&float_data)?;
/// println!("Standard deviation of floats: {}", sd);
///
/// // Handle empty input
/// let empty_data: &[i32] = &[];
/// assert!(std_dev(empty_data).is_err());
/// # Ok::<(), rs_stats::StatsError>(())
/// ```
#[inline]
pub fn std_dev<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    variance(data).map(|x| x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_population_std_dev_integers() {
        let data = vec![1, 2, 3, 4, 5];
        let result = std_dev(&data).unwrap();
        let expected = 2.0_f64.sqrt(); // sqrt of population variance (2.0)
        assert!(
            (result - expected).abs() < EPSILON,
            "Population std_dev for integers should be sqrt(2.0)"
        );
    }

    #[test]
    fn test_population_std_dev_floats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_dev(&data).unwrap();
        let expected = 2.0_f64.sqrt(); // sqrt of population variance (2.0)
        assert!(
            (result - expected).abs() < EPSILON,
            "Population std_dev for floats should be sqrt(2.0)"
        );
    }

    #[test]
    fn test_population_std_dev_mixed_floats() {
        let data = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = std_dev(&data).unwrap();
        let expected = 2.0_f64.sqrt(); // sqrt of population variance (2.0)
        assert!(
            (result - expected).abs() < EPSILON,
            "Population std_dev for mixed floats should be sqrt(2.0)"
        );
    }
}
