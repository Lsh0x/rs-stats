use std::f64::consts::SQRT_2;

use crate::erf;

/// CDF return the CDF for the zscore given
/// https://en.wikipedia.org/wiki/Cumulative_distribution_function#Definition
#[inline]
pub fn normal_cummulative_distrib(z: f64) -> f64 {
    (1.0 + erf(z / SQRT_2)) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    const EPSILON: f64 = 1e-5;

    #[test]
    fn test_cdf_z_zero() {
        let z = 0.0;
        let result = normal_cummulative_distrib(z);
        let expected = 0.5; // CDF(0) = 0.5 for a standard normal distribution
        assert!((result - expected).abs() < EPSILON, "CDF for z = 0.0 should be 0.5");
    }

    #[test]
    fn test_cdf_positive_z() {
        let z = 1.0;
        let result = normal_cummulative_distrib(z);
        let expected = 0.841344746; // CDF(1.0) in a standard normal distribution
        assert!((result - expected).abs() < EPSILON, "CDF for z = 1.0 should match expected");
    }

    #[test]
    fn test_cdf_negative_z() {
        let z = -1.0;
        let result = normal_cummulative_distrib(z);
        let expected = 0.158655254; // CDF(-1.0) in a standard normal distribution
        assert!((result - expected).abs() < EPSILON, "CDF for z = -1.0 should match expected");
    }

    #[test]
    fn test_cdf_large_positive_z() {
        let z = 3.0;
        let result = normal_cummulative_distrib(z);
        let expected = 0.998650102; // CDF(3.0) in a standard normal distribution
        assert!((result - expected).abs() < EPSILON, "CDF for z = 3.0 should match expected");
    }

    #[test]
    fn test_cdf_large_negative_z() {
        let z = -3.0;
        let result = normal_cummulative_distrib(z);
        let expected = 0.001349898; // CDF(-3.0) in a standard normal distribution
        assert!((result - expected).abs() < EPSILON, "CDF for z = -3.0 should match expected");
    }
}

