//! # Standard Deviation
//!
//! Population and sample standard deviation. See [`crate::prob::variance`]
//! for the population-vs-sample convention; this module just takes
//! `sqrt(variance)`.

use crate::error::StatsResult;
use crate::prob::variance::{variance_population, variance_sample};
use num_traits::ToPrimitive;
use std::fmt::Debug;

/// Population standard deviation (`sqrt(variance_population)`).
///
/// Same as `numpy.std(data)` / `numpy.std(data, ddof=0)`.
///
/// # Examples
/// ```
/// use rs_stats::prob::std_dev_population;
/// let s = std_dev_population(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((s - 2.0_f64.sqrt()).abs() < 1e-12);
/// ```
#[inline]
pub fn std_dev_population<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    variance_population(data).map(f64::sqrt)
}

/// Sample standard deviation (`sqrt(variance_sample)`).
///
/// Same as `numpy.std(data, ddof=1)` / `pandas.Series.std()`.
///
/// # Examples
/// ```
/// use rs_stats::prob::std_dev_sample;
/// let s = std_dev_sample(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((s - 2.5_f64.sqrt()).abs() < 1e-12);
/// ```
#[inline]
pub fn std_dev_sample<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    variance_sample(data).map(f64::sqrt)
}

/// Population standard deviation — alias for [`std_dev_population`].
///
/// Kept for v2.x source-compatibility. Use [`std_dev_sample`] when you
/// need the Bessel-corrected estimator.
#[inline]
pub fn std_dev<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    std_dev_population(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_matches_alias() {
        let data = [1.0, 2.5, 3.0, 4.5, 5.0];
        assert_eq!(std_dev(&data).unwrap(), std_dev_population(&data).unwrap());
    }

    #[test]
    fn population_known_value() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((std_dev_population(&data).unwrap() - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn sample_known_value() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((std_dev_sample(&data).unwrap() - 2.5_f64.sqrt()).abs() < 1e-12);
    }
}
