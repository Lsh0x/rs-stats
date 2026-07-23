//! # Descriptive statistics in one call
//!
//! [`describe`] summarises a dataset (count, mean, sample std-dev, min,
//! quartiles, max); [`quantile`] computes any empirical quantile with
//! linear interpolation (numpy's default, "type 7" in the Hyndman-Fan
//! taxonomy).

use crate::error::{StatsError, StatsResult};
use crate::prob::std_dev_sample;
use num_traits::ToPrimitive;

/// Five-number-plus summary of a dataset.
#[derive(Debug, Clone, Copy)]
pub struct Description {
    /// Number of observations.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// **Sample** standard deviation (ddof = 1); `NaN` when `n < 2`.
    pub std_dev: f64,
    /// Minimum.
    pub min: f64,
    /// First quartile (25th percentile).
    pub q1: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Third quartile (75th percentile).
    pub q3: f64,
    /// Maximum.
    pub max: f64,
}

fn to_sorted_f64<T: ToPrimitive>(data: &[T], caller: &str) -> StatsResult<Vec<f64>> {
    if data.is_empty() {
        return Err(StatsError::empty_data(format!(
            "{caller}: data must not be empty"
        )));
    }
    let mut xs: Vec<f64> = data
        .iter()
        .map(|v| v.to_f64())
        .collect::<Option<_>>()
        .ok_or_else(|| {
            StatsError::conversion_error(format!("{caller}: data not convertible to f64"))
        })?;
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(xs)
}

/// Empirical quantile of **already sorted** data (linear interpolation).
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    // numpy default ("linear" / type 7): h = (n−1)·q.
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Empirical quantile `q ∈ [0, 1]` of `data`, with linear interpolation
/// between order statistics (matches `numpy.quantile`'s default).
///
/// # Errors
/// Returns an error on empty data or `q` outside `[0, 1]`.
///
/// # Examples
/// ```
/// use rs_stats::prob::quantile;
///
/// let data = [2.0, 7.0, 4.0, 1.0, 9.0, 3.0, 8.0, 5.0, 6.0, 10.0];
/// assert!((quantile(&data, 0.5).unwrap() - 5.5).abs() < 1e-12);
/// assert!((quantile(&data, 0.9).unwrap() - 9.1).abs() < 1e-12);
/// ```
pub fn quantile<T: ToPrimitive>(data: &[T], q: f64) -> StatsResult<f64> {
    if !(0.0..=1.0).contains(&q) {
        return Err(StatsError::invalid_parameter(format!(
            "quantile: q must be in [0, 1], got {q}"
        )));
    }
    let sorted = to_sorted_f64(data, "quantile")?;
    Ok(quantile_sorted(&sorted, q))
}

/// One-call descriptive summary: count, mean, sample std-dev, min,
/// quartiles and max.
///
/// # Examples
/// ```
/// use rs_stats::prob::describe;
///
/// let data = [12.1, 14.3, 13.8, 15.2, 14.9, 13.1, 14.0, 16.4];
/// let d = describe(&data).unwrap();
/// assert_eq!(d.n, 8);
/// assert!(d.min <= d.q1 && d.q1 <= d.median && d.median <= d.q3 && d.q3 <= d.max);
/// ```
pub fn describe<T: ToPrimitive + std::fmt::Debug>(data: &[T]) -> StatsResult<Description> {
    let sorted = to_sorted_f64(data, "describe")?;
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let std_dev = if n >= 2 {
        std_dev_sample(&sorted)?
    } else {
        f64::NAN
    };
    Ok(Description {
        n,
        mean,
        std_dev,
        min: sorted[0],
        q1: quantile_sorted(&sorted, 0.25),
        median: quantile_sorted(&sorted, 0.5),
        q3: quantile_sorted(&sorted, 0.75),
        max: sorted[n - 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_matches_numpy() {
        // Reference: numpy.quantile (linear interpolation).
        let data = [2.0, 7.0, 4.0, 1.0, 9.0, 3.0, 8.0, 5.0, 6.0, 10.0];
        assert!((quantile(&data, 0.25).unwrap() - 3.25).abs() < 1e-12);
        assert!((quantile(&data, 0.5).unwrap() - 5.5).abs() < 1e-12);
        assert!((quantile(&data, 0.9).unwrap() - 9.1).abs() < 1e-12);
        assert_eq!(quantile(&data, 0.0).unwrap(), 1.0);
        assert_eq!(quantile(&data, 1.0).unwrap(), 10.0);
    }

    #[test]
    fn test_describe_basic() {
        let data = [1, 2, 3, 4, 5];
        let d = describe(&data).unwrap();
        assert_eq!(d.n, 5);
        assert!((d.mean - 3.0).abs() < 1e-12);
        assert!((d.std_dev - 2.5_f64.sqrt()).abs() < 1e-12);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.median, 3.0);
        assert_eq!(d.max, 5.0);
    }

    #[test]
    fn test_describe_errors_and_edge() {
        assert!(describe::<f64>(&[]).is_err());
        assert!(quantile(&[1.0], 1.5).is_err());
        let d = describe(&[42.0]).unwrap();
        assert_eq!(d.median, 42.0);
        assert!(d.std_dev.is_nan());
    }
}
