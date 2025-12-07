//! # Mathematical Constants
//!
//! This module provides precomputed mathematical constants used throughout the library.
//! Using these constants avoids repeated computations and improves performance.

/// Inverse square root of 2π
/// Value: 1.0 / sqrt(2π) ≈ 0.3989422804014327
///
/// Used in normal distribution PDF calculations:
/// PDF(x) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))
///        = (1/σ) * (1/√(2π)) * e^(-z²/2)
///        = (1/σ) * INV_SQRT_2PI * e^(-z²/2)
pub const INV_SQRT_2PI: f64 = 0.3989422804014327;

/// Square root of 2π
/// Value: sqrt(2π) ≈ 2.5066282746310002
///
/// Used in normal distribution calculations.
pub const SQRT_2PI: f64 = 2.5066282746310002;

/// Square root of 2
/// Value: sqrt(2) ≈ 1.4142135623730951
///
/// Re-exported from std::f64::consts::SQRT_2 for convenience.
/// Used in normal distribution CDF calculations.
pub use std::f64::consts::SQRT_2;

/// Pi (π)
/// Re-exported from std::f64::consts::PI for convenience.
pub use std::f64::consts::PI;

/// Euler's number (e)
/// Re-exported from std::f64::consts::E for convenience.
pub use std::f64::consts::E;

/// Natural logarithm of 2π
/// Value: ln(2π) ≈ 1.8378770664093456
///
/// Used in statistical calculations, particularly in t-tests.
pub const LN_2PI: f64 = 1.8378770664093456;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E as STD_E, PI as STD_PI, SQRT_2 as STD_SQRT_2};

    #[test]
    fn test_inv_sqrt_2pi() {
        let expected = 1.0 / (2.0 * STD_PI).sqrt();
        assert!(
            (INV_SQRT_2PI - expected).abs() < 1e-15,
            "INV_SQRT_2PI should equal 1/sqrt(2π)"
        );
    }

    #[test]
    fn test_sqrt_2pi() {
        let expected = (2.0 * STD_PI).sqrt();
        assert!(
            (SQRT_2PI - expected).abs() < 1e-15,
            "SQRT_2PI should equal sqrt(2π)"
        );
    }

    #[test]
    fn test_constants_relationship() {
        // Verify INV_SQRT_2PI * SQRT_2PI ≈ 1.0
        assert!(
            (INV_SQRT_2PI * SQRT_2PI - 1.0).abs() < 1e-15,
            "INV_SQRT_2PI * SQRT_2PI should equal 1.0"
        );
    }

    #[test]
    fn test_reexported_constants() {
        assert_eq!(SQRT_2, STD_SQRT_2);
        assert_eq!(PI, STD_PI);
        assert_eq!(E, STD_E);
    }
}
