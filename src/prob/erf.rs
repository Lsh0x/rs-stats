//! # Error Function (erf)
//!
//! This module implements the error function, a special function that occurs in probability,
//! statistics, and partial differential equations.
//!
//! ## Mathematical Definition
//! The error function is defined as:
//!
//! erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
//!
//! ## Key Properties
//! - erf(-x) = -erf(x) (odd function)
//! - erf(0) = 0
//! - erf(∞) = 1
//! - erf(-∞) = -1
//!
//! ## Implementation Details
//! Uses Abramowitz and Stegun formula 7.1.26 for approximation
//! with maximum error of 1.5 × 10⁻⁷

/// Calculate the error function (erf) of a value
///
/// The error function is related to the normal distribution and is used
/// in probability calculations.
///
/// # Arguments
/// * `x` - The value at which to evaluate the error function
///
/// # Returns
/// The value of the error function at x
///
/// # Examples
/// ```
/// use rs_stats::prob::erf;
///
/// // Calculate erf(1.0)
/// let result = erf(1.0);
/// assert!((result - 0.8427006897475899).abs() < 1e-8);
///
/// // Verify symmetry property
/// assert!((erf(1.0) + erf(-1.0)).abs() < 1e-8);
/// ```
pub fn erf(x: f64) -> f64 {
    // Special case: return exactly 0.0 when x is 0.0
    if x == 0.0 {
        return 0.0;
    }

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Constants for the approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    // Abramowitz and Stegun formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_special_cases() {
        assert!((erf(f64::INFINITY) - 1.0).abs() < 1e-10);
        assert!((erf(f64::NEG_INFINITY) + 1.0).abs() < 1e-10);
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn test_erf_against_known_values() {
        let test_cases = vec![
            (-3.0, -0.999977909503),
            (-2.0, -0.995322265019),
            (-1.0, -0.842700792950),
            (0.0, 0.0),
            (0.5, 0.520499877813),
            (1.0, 0.842700792950),
            (2.0, 0.995322265019),
            (3.0, 0.999977909503),
        ];

        for (x, expected) in test_cases {
            let actual = erf(x);
            assert!(
                (actual - expected).abs() < 1e-6,
                "For x = {}, expected {}, but got {}",
                x,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_erf_symmetry() {
        let x = 0.7;
        let actual = erf(x) + erf(-x);
        assert!(
            actual.abs() < 1e-10,
            "erf(x) + erf(-x) should be 0.0, but got {}",
            actual
        );
    }

    #[test]
    fn test_erf_limits() {
        // Test erf approaching its limits
        assert!((erf(10.0) - 1.0).abs() < 1e-15); // erf(x) -> 1 as x -> +inf
        assert!((erf(-10.0) + 1.0).abs() < 1e-15); // erf(x) -> -1 as x -> -inf
    }

    #[test]
    fn test_erf_large_negative() {
        let x = -8.0;
        let actual = erf(x);
        assert!(
            (actual + 1.0).abs() < 1e-10,
            "For large negative x, erf(x) should be close to -1.0, but got {}",
            actual
        );
    }
}
