use num_traits::NumCast;
/// Provides numerical utility functions for statistical calculations.
use std::fmt::Debug;

/// Computes the natural logarithm of x, handling edge cases safely.
///
/// # Arguments
/// * `x` - The input number.
///
/// # Returns
/// * `Result<f64>` - The natural logarithm of x, or an error message if x is invalid.
///
/// # Errors
/// Returns an error if x is less than or equal to 0.
pub fn safe_log(x: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Err("Logarithm is only defined for positive numbers.".to_string())
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
}
