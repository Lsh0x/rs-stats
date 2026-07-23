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

use crate::error::StatsResult;
use crate::prob::{std_dev_population, std_dev_sample};
use num_traits::ToPrimitive;

/// Calculate the standard error of the mean (SEM) of a dataset.
///
/// Uses the **sample** standard deviation (ddof = 1), the standard SEM
/// convention (matches `scipy.stats.sem`). The population variant is
/// available as [`std_err_population`].
///
/// # Arguments
/// * `data` - A slice of numeric values implementing `ToPrimitive`
///
/// # Returns
/// * `StatsResult<f64>` - The standard error, or an error if the input is invalid
///
/// # Errors
/// Returns `StatsError::EmptyData` if the input slice is empty.
/// Returns `StatsError::InvalidInput` if the slice has fewer than 2 elements
/// (the sample standard deviation needs n ≥ 2).
/// Returns `StatsError::ConversionError` if any value cannot be converted to f64.
///
/// # Examples
/// ```
/// use rs_stats::prob::std_err;
///
/// // Calculate standard error for a dataset (sample convention, like scipy.stats.sem)
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let se = std_err(&data)?;
/// assert!((se - 0.7071067811865476).abs() < 1e-9);
///
/// // Handle empty input
/// let empty_data: &[f64] = &[];
/// assert!(std_err(empty_data).is_err());
/// # Ok::<(), rs_stats::StatsError>(())
/// ```
#[inline]
pub fn std_err<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + std::fmt::Debug,
{
    std_dev_sample(data).map(|std| std / (data.len() as f64).sqrt())
}

/// Standard error of the mean using the **population** standard deviation
/// (ddof = 0).
///
/// This was the (undocumented) behaviour of [`std_err`] before v3.1; prefer
/// [`std_err`] unless you specifically have the full population.
#[inline]
pub fn std_err_population<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + std::fmt::Debug,
{
    std_dev_population(data).map(|std| std / (data.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_std_err_integers() {
        // Dataset: [1, 2, 3, 4, 5]
        // Sample std-dev (ddof=1) = sqrt(2.5) ≈ 1.5811388
        // SEM = 1.5811388 / sqrt(5) ≈ 0.7071068 (matches scipy.stats.sem)
        let data = vec![1, 2, 3, 4, 5];
        let result = std_err(&data).unwrap();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result - expected).abs() < EPSILON,
            "Standard error should be approximately 0.7071068"
        );
    }

    #[test]
    fn test_std_err_floats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_err(&data).unwrap();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result - expected).abs() < EPSILON,
            "Standard error for floats should be approximately 0.7071068"
        );
    }

    #[test]
    fn test_std_err_population_variant() {
        // Population convention: pop std-dev = sqrt(2) ≈ 1.4142136,
        // SEM_pop = 1.4142136 / sqrt(5) ≈ 0.6324555.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_err_population(&data).unwrap();
        let expected = 0.6324555320336759;
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_std_err_single_element() {
        // The sample standard deviation is undefined for n = 1 —
        // a single point carries no information about spread.
        let data = vec![5];
        assert!(std_err(&data).is_err());
    }

    #[test]
    fn test_std_err_empty() {
        // Empty dataset: []
        // There should be no standard error, result should be an error
        let data: Vec<i32> = vec![];
        let result = std_err(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::StatsError::EmptyData { .. }
        ));
    }
}
