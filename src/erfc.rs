use crate::erf;

/// erfc returns the error function integrated between x and infinity
/// called complementary error function
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
