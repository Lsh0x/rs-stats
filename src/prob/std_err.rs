//! # Standard Error Calculation
//!
//! This module implements the standard error calculation, which measures
//! the precision of the sample mean as an estimate of the population mean.
//!
//! ## Mathematical Definition
//! The standard error is defined as:
//!
//! SE = σ / √n
//!
//! where:
//! - σ is the sample standard deviation
//! - n is the sample size
//!
//! ## Key Properties
//! - Decreases as sample size increases
//! - Measures the variability of the sample mean
//! - Used in confidence intervals and hypothesis testing

use num_traits::ToPrimitive;
use crate::prob::std_dev::std_dev;

/// Calculate the standard error of a dataset
///
/// The standard error quantifies the uncertainty in the sample mean
/// as an estimate of the population mean.
///
/// # Arguments
/// * `data` - A slice of numeric values implementing `ToPrimitive`
///
/// # Returns
/// * `Some(f64)` - The standard error if the input slice is non-empty
/// * `None` - If the input slice is empty
///
/// # Examples
/// ```
/// use rs_stats::prob::std_err;
///
/// // Calculate standard error for a dataset
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let se = std_err(&data).unwrap();
/// assert!((se - 0.632455532).abs() < 1e-9);
///
/// // Handle empty input
/// let empty_data: &[f64] = &[];
/// assert!(std_err(empty_data).is_none());
/// ```
#[inline]
pub fn std_err<T>(data: &[T]) -> Option<f64>
where
    T: ToPrimitive + std::fmt::Debug,
{
    std_dev(data).map(|std| std / (data.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_std_err_integers() {
        // Dataset: [1, 2, 3, 4, 5]
        // Standard deviation of [1, 2, 3, 4, 5] is 1.414213562 (approx)
        // Standard error should be std_dev / sqrt(n) = 1.414213562 / sqrt(5) = 0.632455532 (approx)
        let data = vec![1, 2, 3, 4, 5];
        let result = std_err(&data);
        let expected = 0.632455532; // Calculated value of the standard error
        assert!(
            (result.unwrap() - expected).abs() < EPSILON,
            "Standard error should be approximately 0.632455532"
        );
    }

    #[test]
    fn test_std_err_floats() {
        // Dataset: [1.0, 2.0, 3.0, 4.0, 5.0]
        // Standard deviation of [1.0, 2.0, 3.0, 4.0, 5.0] is the same as for integers
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_err(&data);
        let expected = 0.632455532;
        assert!(
            (result.unwrap() - expected).abs() < EPSILON,
            "Standard error for floats should be approximately 0.632455532"
        );
    }

    #[test]
    fn test_std_err_single_element() {
        // Dataset with only one element: [5]
        // Standard deviation is 0, and thus standard error should also be 0
        let data = vec![5];
        let result = std_err(&data);
        let expected = 0.0;
        assert_eq!(
            result,
            Some(expected),
            "Standard error for a single element should be 0.0"
        );
    }

    #[test]
    fn test_std_err_empty() {
        // Empty dataset: []
        // There should be no standard error, result should be None
        let data: Vec<i32> = vec![];
        let result = std_err(&data);
        assert_eq!(
            result, None,
            "Standard error for an empty dataset should be None"
        );
    }
}
