//! # Variance Calculation
//!
//! Population and sample variance over numeric slices, single-pass online
//! (Welford 1962) — never walks `data` twice and never allocates.
//!
//! ## Convention
//!
//! `rs-stats` follows the explicit-naming convention from numpy and scipy:
//!
//! - [`variance`] / [`variance_population`] — divide by `n` (population
//!   variance, **MLE** estimator). Equivalent to `numpy.var(data)` /
//!   `numpy.var(data, ddof=0)`.
//! - [`variance_sample`] — divide by `n - 1` (sample variance,
//!   Bessel-corrected unbiased estimator). Equivalent to
//!   `numpy.var(data, ddof=1)` / `pandas.Series.var()`.
//!
//! [`variance`] and [`variance_population`] are the same function — the
//! shorter name is kept because it matches the previous (v2.x) behaviour
//! and is what every distribution's [`Distribution::variance`] reports.
//! When in doubt, prefer the explicit name.
//!
//! [`Distribution::variance`]: crate::distributions::traits::Distribution::variance

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;
use std::fmt::Debug;

/// Population variance (divide by `n`, MLE).
///
/// Same as `numpy.var(data)` / `numpy.var(data, ddof=0)`.
///
/// # Arguments
/// * `data` - A slice of numeric values implementing `ToPrimitive`.
///
/// # Errors
/// * [`StatsError::EmptyData`] if `data` is empty.
/// * [`StatsError::ConversionError`] if a value cannot be converted to `f64`.
///
/// # Examples
/// ```
/// use rs_stats::prob::variance_population;
/// let v = variance_population(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((v - 2.0).abs() < 1e-12); // Σ(xᵢ−μ)² / n = 10/5
/// ```
#[inline]
pub fn variance_population<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    let (n, _, m2) = welford_pass(data, "prob::variance_population")?;
    Ok(m2 / n)
}

/// Sample variance (divide by `n - 1`, Bessel-corrected unbiased estimator).
///
/// Same as `numpy.var(data, ddof=1)` / `pandas.Series.var()`.
///
/// # Errors
/// * [`StatsError::EmptyData`] if `data` is empty.
/// * [`StatsError::InvalidInput`] if `data.len() < 2` (sample variance is
///   undefined for n = 1).
/// * [`StatsError::ConversionError`] if a value cannot be converted to `f64`.
///
/// # Examples
/// ```
/// use rs_stats::prob::variance_sample;
/// let v = variance_sample(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((v - 2.5).abs() < 1e-12); // Σ(xᵢ−μ)² / (n−1) = 10/4
/// ```
#[inline]
pub fn variance_sample<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    let (n, _, m2) = welford_pass(data, "prob::variance_sample")?;
    if n < 2.0 {
        return Err(StatsError::invalid_input(
            "prob::variance_sample: need at least 2 observations for sample variance",
        ));
    }
    Ok(m2 / (n - 1.0))
}

/// Population variance — alias for [`variance_population`].
///
/// Kept under the short name for ergonomics and v2.x source-compatibility;
/// every [`Distribution::variance`] in this crate reports the same
/// (population) quantity. Use [`variance_sample`] when you need the
/// Bessel-corrected estimator.
///
/// [`Distribution::variance`]: crate::distributions::traits::Distribution::variance
///
/// # Examples
/// ```
/// use rs_stats::prob::variance;
/// let v = variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((v - 2.0).abs() < 1e-12);
/// ```
#[inline]
pub fn variance<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    variance_population(data)
}

/// Single-pass Welford pass: returns `(n, mean, m2)` over `data`,
/// converting each element to f64 with the supplied error context.
#[inline]
fn welford_pass<T>(data: &[T], ctx: &str) -> StatsResult<(f64, f64, f64)>
where
    T: ToPrimitive + Debug,
{
    if data.is_empty() {
        return Err(StatsError::empty_data(format!(
            "{ctx}: cannot compute variance of empty dataset"
        )));
    }
    let mut n = 0.0_f64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for (i, x) in data.iter().enumerate() {
        let value = x.to_f64().ok_or_else(|| {
            StatsError::conversion_error(format!(
                "{ctx}: failed to convert value at index {i} to f64"
            ))
        })?;
        n += 1.0;
        let delta = value - mean;
        mean += delta / n;
        m2 += delta * (value - mean);
    }
    Ok((n, mean, m2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_matches_legacy_alias() {
        let data = [1.0, 2.5, 3.0, 4.5, 5.0];
        assert_eq!(
            variance(&data).unwrap(),
            variance_population(&data).unwrap()
        );
    }

    #[test]
    fn test_population_known_value() {
        // Mean = 3, Σ(xᵢ−μ)² = 10, n = 5 → 10/5 = 2
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance_population(&data).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sample_known_value() {
        // Same numerator, n - 1 = 4 → 10/4 = 2.5
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance_sample(&data).unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_population_single_value_is_zero() {
        let data = [5];
        assert_eq!(variance_population(&data).unwrap(), 0.0);
    }

    #[test]
    fn test_sample_single_value_errors() {
        let data = [5];
        assert!(matches!(
            variance_sample(&data),
            Err(StatsError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_empty_errors() {
        let data: &[f64] = &[];
        assert!(matches!(variance(data), Err(StatsError::EmptyData { .. })));
        assert!(matches!(
            variance_sample(data),
            Err(StatsError::EmptyData { .. })
        ));
    }

    #[test]
    fn test_identical_values_zero_variance() {
        let data = [2.0; 10];
        assert_eq!(variance_population(&data).unwrap(), 0.0);
        assert_eq!(variance_sample(&data).unwrap(), 0.0);
    }
}
