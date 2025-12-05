//! # Error Types
//!
//! This module defines the error types used throughout the rs-stats library.
//! All errors are structured and provide context about what went wrong.

use thiserror::Error;

/// Main error type for the rs-stats library
///
/// This enum represents all possible errors that can occur in the library.
/// Each variant includes a message providing context about the error.
///
/// # Examples
///
/// ```rust
/// use rs_stats::error::{StatsError, StatsResult};
///
/// fn example() -> StatsResult<f64> {
///     Err(StatsError::InvalidInput {
///         message: "Value must be positive".to_string(),
///     })
/// }
/// ```
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Invalid input parameters provided to a function
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Human-readable error message
        message: String,
    },

    /// Type conversion failure
    #[error("Conversion error: {message}")]
    ConversionError {
        /// Human-readable error message
        message: String,
    },

    /// Empty data provided when data is required
    #[error("Empty data: {message}")]
    EmptyData {
        /// Human-readable error message
        message: String,
    },

    /// Dimension mismatch between arrays/vectors
    #[error("Dimension mismatch: {message}")]
    DimensionMismatch {
        /// Human-readable error message
        message: String,
    },

    /// Numerical computation error (overflow, underflow, NaN, etc.)
    #[error("Numerical error: {message}")]
    NumericalError {
        /// Human-readable error message
        message: String,
    },

    /// Model not fitted/trained before use
    #[error("Model not fitted: {message}")]
    NotFitted {
        /// Human-readable error message
        message: String,
    },

    /// Invalid parameter value
    #[error("Invalid parameter: {message}")]
    InvalidParameter {
        /// Human-readable error message
        message: String,
    },

    /// Index out of bounds
    #[error("Index out of bounds: {message}")]
    IndexOutOfBounds {
        /// Human-readable error message
        message: String,
    },

    /// Division by zero or similar mathematical error
    #[error("Mathematical error: {message}")]
    MathematicalError {
        /// Human-readable error message
        message: String,
    },
}

/// Convenience type alias for Result with StatsError
///
/// This is the standard return type for functions that can fail in the rs-stats library.
///
/// # Examples
///
/// ```rust
/// use rs_stats::error::StatsResult;
///
/// fn might_fail() -> StatsResult<f64> {
///     Ok(42.0)
/// }
/// ```
pub type StatsResult<T> = Result<T, StatsError>;

impl StatsError {
    /// Create an InvalidInput error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        StatsError::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a ConversionError
    pub fn conversion_error<S: Into<String>>(message: S) -> Self {
        StatsError::ConversionError {
            message: message.into(),
        }
    }

    /// Create an EmptyData error
    pub fn empty_data<S: Into<String>>(message: S) -> Self {
        StatsError::EmptyData {
            message: message.into(),
        }
    }

    /// Create a DimensionMismatch error
    pub fn dimension_mismatch<S: Into<String>>(message: S) -> Self {
        StatsError::DimensionMismatch {
            message: message.into(),
        }
    }

    /// Create a NumericalError
    pub fn numerical_error<S: Into<String>>(message: S) -> Self {
        StatsError::NumericalError {
            message: message.into(),
        }
    }

    /// Create a NotFitted error
    pub fn not_fitted<S: Into<String>>(message: S) -> Self {
        StatsError::NotFitted {
            message: message.into(),
        }
    }

    /// Create an InvalidParameter error
    pub fn invalid_parameter<S: Into<String>>(message: S) -> Self {
        StatsError::InvalidParameter {
            message: message.into(),
        }
    }

    /// Create an IndexOutOfBounds error
    pub fn index_out_of_bounds<S: Into<String>>(message: S) -> Self {
        StatsError::IndexOutOfBounds {
            message: message.into(),
        }
    }

    /// Create a MathematicalError
    pub fn mathematical_error<S: Into<String>>(message: S) -> Self {
        StatsError::MathematicalError {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Display implementation for all variants
    #[test]
    fn test_all_variants_display() {
        let cases = vec![
            (StatsError::invalid_input("msg"), "Invalid input: msg"),
            (StatsError::conversion_error("msg"), "Conversion error: msg"),
            (StatsError::empty_data("msg"), "Empty data: msg"),
            (
                StatsError::dimension_mismatch("msg"),
                "Dimension mismatch: msg",
            ),
            (StatsError::numerical_error("msg"), "Numerical error: msg"),
            (StatsError::not_fitted("msg"), "Model not fitted: msg"),
            (
                StatsError::invalid_parameter("msg"),
                "Invalid parameter: msg",
            ),
            (
                StatsError::index_out_of_bounds("msg"),
                "Index out of bounds: msg",
            ),
            (
                StatsError::mathematical_error("msg"),
                "Mathematical error: msg",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected, "Display format mismatch");
        }
    }

    /// Test equality between errors
    #[test]
    fn test_error_equality() {
        let err1 = StatsError::invalid_input("message");
        let err2 = StatsError::invalid_input("message");
        let err3 = StatsError::invalid_input("different");
        let err4 = StatsError::conversion_error("message");

        assert_eq!(err1, err2, "Same variant and message should be equal");
        assert_ne!(err1, err3, "Different messages should not be equal");
        assert_ne!(err1, err4, "Different variants should not be equal");
    }

    /// Test Clone implementation
    #[test]
    fn test_error_clone() {
        let err = StatsError::conversion_error("test");
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    /// Test StatsResult type alias
    #[test]
    fn test_stats_result() {
        let ok: StatsResult<f64> = Ok(42.0);
        assert_eq!(ok.unwrap(), 42.0);

        let err: StatsResult<f64> = Err(StatsError::invalid_input("test"));
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), StatsError::invalid_input("test"));
    }

    /// Test helper methods return correct variants
    #[test]
    fn test_helper_methods() {
        assert!(matches!(
            StatsError::invalid_input("msg"),
            StatsError::InvalidInput { .. }
        ));
        assert!(matches!(
            StatsError::conversion_error("msg"),
            StatsError::ConversionError { .. }
        ));
        assert!(matches!(
            StatsError::empty_data("msg"),
            StatsError::EmptyData { .. }
        ));
        assert!(matches!(
            StatsError::dimension_mismatch("msg"),
            StatsError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            StatsError::numerical_error("msg"),
            StatsError::NumericalError { .. }
        ));
        assert!(matches!(
            StatsError::not_fitted("msg"),
            StatsError::NotFitted { .. }
        ));
        assert!(matches!(
            StatsError::invalid_parameter("msg"),
            StatsError::InvalidParameter { .. }
        ));
        assert!(matches!(
            StatsError::index_out_of_bounds("msg"),
            StatsError::IndexOutOfBounds { .. }
        ));
        assert!(matches!(
            StatsError::mathematical_error("msg"),
            StatsError::MathematicalError { .. }
        ));
    }

    /// Test Into<String> conversion for helper methods
    #[test]
    fn test_into_string_conversion() {
        // Test with &str
        let err1 = StatsError::invalid_input("string slice");
        assert_eq!(err1.to_string(), "Invalid input: string slice");

        // Test with String
        let err2 = StatsError::invalid_input("owned string".to_string());
        assert_eq!(err2.to_string(), "Invalid input: owned string");
    }

    /// Test error propagation with ? operator
    #[test]
    fn test_error_propagation() {
        fn might_fail() -> StatsResult<f64> {
            Err(StatsError::invalid_input("test"))
        }

        fn propagate() -> StatsResult<f64> {
            might_fail()?;
            Ok(42.0)
        }

        let result = propagate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    /// Test pattern matching on error variants
    #[test]
    fn test_pattern_matching() {
        let err = StatsError::conversion_error("failed");

        match err {
            StatsError::ConversionError { message } => {
                assert_eq!(message, "failed");
            }
            _ => panic!("Wrong variant matched"),
        }
    }

    /// Test Debug implementation
    #[test]
    fn test_debug_implementation() {
        let err = StatsError::invalid_input("test");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidInput"));
        assert!(debug_str.contains("test"));
    }
}
