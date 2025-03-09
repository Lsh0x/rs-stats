//! # Complementary Error Function (erfc)
//!
//! This module implements the complementary error function, which is related to
//! the normal distribution and is used in probability calculations.
//!
//! ## Mathematical Definition
//! The complementary error function is defined as:
//!
//! erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
//!
//! ## Key Properties
//! - erfc(-∞) = 2
//! - erfc(0) = 1
//! - erfc(∞) = 0
//! - erfc(-x) = 2 - erfc(x)
//!
//! ## Relationship to Normal Distribution
//! For a standard normal distribution N(0,1):
//! P(X > x) = 0.5 * erfc(x/√2)

use crate::prob::erf::erf;

/// Calculate the complementary error function (erfc) of a value
///
/// The complementary error function is particularly useful for calculating
/// tail probabilities of the normal distribution.
///
/// # Arguments
/// * `x` - The value at which to evaluate the complementary error function
///
/// # Returns
/// The value of the complementary error function at x
///
/// # Examples
/// ```
/// use rs_stats::prob::erfc;
///
/// // Calculate erfc(1.0)
/// let result = erfc(1.0);
/// assert!((result - 0.15729931025241006).abs() < 1e-8);
///
/// // Verify relationship to normal distribution
/// let p = 0.5 * erfc(1.0 / 2.0f64.sqrt());
/// assert!((p - 0.15865526383236372).abs() < 1e-8); // P(X > 1) for N(0,1)
/// ```
#[inline]
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-5;

    #[test]
    fn test_erfc_zero() {
        // erfc(0) should be 1.0 because erf(0) is 0
        let result = erfc(0.0);
        let expected = 1.0;
        assert!((result - expected).abs() < EPSILON, "erfc(0) should be 1.0");
    }

    #[test]
    fn test_erfc_positive_value() {
        // Testing erfc(1.0)
        // Known value: erfc(1.0) is approximately 0.157299207
        let result = erfc(1.0);
        let expected = 0.157299207;
        assert!(
            (result - expected).abs() < EPSILON,
            "erfc(1.0) should be approximately 0.157299207"
        );
    }

    #[test]
    fn test_erfc_negative_value() {
        // Testing erfc(-1.0)
        // Known value: erfc(-1.0) is approximately 1.842700792
        let result = erfc(-1.0);
        let expected = 1.842700792;
        assert!(
            (result - expected).abs() < EPSILON,
            "erfc(-1.0) should be approximately 1.842700792"
        );
    }

    #[test]
    fn test_erfc_large_positive_value() {
        // Testing erfc(3.0)
        // Known value: erfc(3.0) is approximately 0.0000220905
        let result = erfc(3.0);
        let expected = 0.0000220905;
        assert!(
            (result - expected).abs() < EPSILON,
            "erfc(3.0) should be approximately 0.0000220905 got {:?}",
            result
        );
    }

    #[test]
    fn test_erfc_large_negative_value() {
        // Testing erfc(-3.0)
        // Known value: erfc(-3.0) is approximately 1.99997791
        let result = erfc(-3.0);
        let expected = 1.99997791;
        assert!(
            (result - expected).abs() < EPSILON,
            "erfc(-3.0) should be approximately 1.99997791"
        );
    }
}
