use num_traits::NumCast;
/// Provides numerical utility functions for statistical calculations.
use std::fmt::Debug;

use crate::error::{StatsResult, StatsError};

/// Computes the natural logarithm of x, handling edge cases safely.
///
/// # Arguments
/// * `x` - The input number.
///
/// # Returns
/// * `StatsResult<f64>` - The natural logarithm of x, or an error if x is invalid.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if x is less than or equal to 0.
///
/// # Examples
/// ```
/// use rs_stats::utils::safe_log;
///
/// let result = safe_log(2.71828).unwrap();
/// assert!((result - 1.0).abs() < 1e-5);
///
/// // Error case
/// let result = safe_log(0.0);
/// assert!(result.is_err());
/// ```

pub fn safe_log(x: f64) -> StatsResult<f64> {
    if x <= 0.0 {
        Err(StatsError::invalid_input(
            "Logarithm is only defined for positive numbers.",
        ))
    } else {
        Ok(x.ln())
    }
}

/// Check if two numeric values are approximately equal within a specified epsilon.
///
/// This function works with any floating point or integer type that can be converted to a float.
/// For integer types, it converts them to f64 for comparison.
///
/// # Arguments
/// * `a` - First value to compare
/// * `b` - Second value to compare
/// * `epsilon` - Tolerance for equality comparison (defaults to 1e-10 if not specified)
///
/// # Returns
/// * `bool` - True if the values are approximately equal, false otherwise
///
pub fn approx_equal<T, U>(a: T, b: U, epsilon: Option<f64>) -> bool
where
    T: NumCast + Copy + Debug,
    U: NumCast + Copy + Debug,
{
    // Convert to f64 for comparison
    let a_f64 = match T::to_f64(&a) {
        Some(val) => val,
        None => return false, // Can't compare if conversion fails
    };

    let b_f64 = match U::to_f64(&b) {
        Some(val) => val,
        None => return false, // Can't compare if conversion fails
    };
    let eps = epsilon.unwrap_or(1e-10);

    // Handle special casesc
    if a_f64.is_nan() || b_f64.is_nan() {
        return false;
    }

    if a_f64.is_infinite() && b_f64.is_infinite() {
        return (a_f64 > 0.0 && b_f64 > 0.0) || (a_f64 < 0.0 && b_f64 < 0.0);
    }

    // Calculate absolute and relative differences
    let abs_diff = (a_f64 - b_f64).abs();

    // For values close to zero, use absolute difference
    if a_f64.abs() < eps || b_f64.abs() < eps {
        return abs_diff <= eps;
    }

    // Otherwise use relative difference
    let rel_diff = abs_diff / f64::max(a_f64.abs(), b_f64.abs());
    rel_diff <= eps
}

/// Simpler interface for approximate equality with default epsilon
pub fn approx_eq<T, U>(a: T, b: U) -> bool
where
    T: NumCast + Copy + Debug,
    U: NumCast + Copy + Debug,
{
    approx_equal(a, b, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_equality() {
        assert!(approx_equal(1.0, 1.0, None));
        assert!(approx_equal(1.0, 1.0000000001, Some(1e-9)));
        assert!(!approx_equal(1.0, 1.0000000001, Some(1e-10)));
    }

    #[test]
    fn test_integer_equality() {
        assert!(approx_equal(1i32, 1i32, None));
        assert!(approx_equal(1000i32, 1000, None));
        assert!(approx_equal(1000u64, 1000.0001, Some(1e-6)));
        assert!(!approx_equal(1000i32, 1001i32, None));
    }

    #[test]
    fn test_mixed_type_equality() {
        assert!(approx_equal(1i32, 1.0f64, None));
        assert!(approx_equal(1000u16, 1000.0f32, None));
        assert!(approx_equal(0i8, 0.0, None));
        assert!(!approx_equal(5u8, 5.1f64, None));
    }

    #[test]
    fn test_edge_cases() {
        assert!(!approx_equal(f64::NAN, f64::NAN, None));
        assert!(approx_equal(f64::INFINITY, f64::INFINITY, None));
        assert!(approx_equal(f64::NEG_INFINITY, f64::NEG_INFINITY, None));
        assert!(!approx_equal(f64::INFINITY, f64::NEG_INFINITY, None));
        assert!(!approx_equal(f64::INFINITY, 1e100, None));
    }

    #[test]
    fn test_near_zero() {
        assert!(approx_equal(0.0, 1e-11, None));
        assert!(!approx_equal(0.0, 1e-9, None));
    }

    // Note: Testing conversion failures in approx_equal is difficult because
    // NumCast::to_f64() for standard numeric types always succeeds.
    // The conversion failure path is mainly for custom types that don't implement NumCast properly.
    // However, we can test the edge cases that are testable.

    #[test]
    fn test_approx_equal_infinity_combinations() {
        // Test all infinity combinations
        assert!(approx_equal(f64::INFINITY, f64::INFINITY, None));
        assert!(approx_equal(f64::NEG_INFINITY, f64::NEG_INFINITY, None));
        assert!(!approx_equal(f64::INFINITY, f64::NEG_INFINITY, None));
        assert!(!approx_equal(f64::NEG_INFINITY, f64::INFINITY, None));
        assert!(!approx_equal(f64::INFINITY, 0.0, None));
        assert!(!approx_equal(f64::NEG_INFINITY, 0.0, None));
    }

    #[test]
    fn test_approx_equal_nan_combinations() {
        // Test NaN combinations
        assert!(!approx_equal(f64::NAN, f64::NAN, None));
        assert!(!approx_equal(f64::NAN, 0.0, None));
        assert!(!approx_equal(0.0, f64::NAN, None));
        assert!(!approx_equal(f64::NAN, f64::INFINITY, None));
        assert!(!approx_equal(f64::INFINITY, f64::NAN, None));
    }

    #[test]
    fn test_approx_equal_relative_difference() {
        // Test relative difference calculation (for values not near zero)
        // Relative diff = |1000.0 - 1000.1| / max(|1000.0|, |1000.1|) = 0.1 / 1000.1 ≈ 0.0001 < 1e-3
        assert!(approx_equal(1000.0, 1000.1, Some(1e-3)));
        // Relative diff = |1000.0 - 1001.0| / max(|1000.0|, |1001.0|) = 1.0 / 1001.0 ≈ 0.001 = 1e-3
        // Since relative_diff <= eps (1e-3), they should be equal
        assert!(approx_equal(1000.0, 1001.0, Some(1e-3)));
        // But with stricter epsilon, they should not be equal
        assert!(!approx_equal(1000.0, 1001.0, Some(1e-4)));
    }

    #[test]
    fn test_approx_equal_absolute_difference_near_zero() {
        // Test absolute difference calculation (for values near zero)
        assert!(approx_equal(1e-11, 0.0, None));
        assert!(approx_equal(0.0, 1e-11, None));
        assert!(!approx_equal(1e-9, 0.0, None));
    }

    #[test]
    fn test_safe_log_positive() {
        let result = safe_log(1.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_safe_log_zero() {
        let result = safe_log(0.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::InvalidInput { .. }));
    }

    #[test]
    fn test_safe_log_negative() {
        let result = safe_log(-1.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::InvalidInput { .. }));
    }

    #[test]
    fn test_safe_log_known_value() {
        // ln(e) = 1
        let result = safe_log(std::f64::consts::E);
        assert!(result.is_ok());
        assert!((result.unwrap() - 1.0).abs() < 1e-10);
    }
}
