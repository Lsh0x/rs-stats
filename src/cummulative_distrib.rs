use std::f64::consts::SQRT_2;

use crate::erf;
use crate::z_score;


/// CDF return the CDF using the mean and the standard deviation given
/// https://en.wikipedia.org/wiki/Cumulative_distribution_function#Definition
#[inline]
pub fn cummulative_distrib(x: f64, avg: f64, stddev: f64) -> f64 {
    (1.0 + erf(z_score(x, avg, stddev) / SQRT_2)) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-5;

    #[test]
    fn test_cdf_standard_normal_at_mean() {
        // For a standard normal distribution (avg = 0, stddev = 1)
        // The CDF at the mean (0) should be 0.5
        let x = 0.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cummulative_distrib(x, avg, stddev);
        let expected = 0.5;
        assert!((result - expected).abs() < EPSILON, "CDF at the mean should be 0.5");
    }

    #[test]
    fn test_cdf_standard_normal_positive() {
        // For a standard normal distribution (avg = 0, stddev = 1)
        // The CDF at 1.0 (z = 1.0) is approximately 0.841344746
        let x = 1.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cummulative_distrib(x, avg, stddev);
        let expected = 0.841344746;
        assert!((result - expected).abs() < EPSILON, "CDF for z = 1.0 should match expected");
    }

    #[test]
    fn test_cdf_standard_normal_negative() {
        // For a standard normal distribution (avg = 0, stddev = 1)
        // The CDF at -1.0 (z = -1.0) is approximately 0.158655254
        let x = -1.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cummulative_distrib(x, avg, stddev);
        let expected = 0.158655254;
        assert!((result - expected).abs() < EPSILON, "CDF for z = -1.0 should match expected");
    }

    #[test]
    fn test_cdf_non_standard_distribution() {
        // For a normal distribution with avg = 10, stddev = 2
        // We can compute the CDF for x = 12, which should give the same result as z = 1.0 for a standard normal distribution
        let x = 12.0;
        let avg = 10.0;
        let stddev = 2.0;
        let result = cummulative_distrib(x, avg, stddev);
        let expected = 0.841344746;  // CDF for z = 1.0 in standard normal
        assert!((result - expected).abs() < EPSILON, "CDF for x = 12 in normal distribution with mean 10 and stddev 2 should match expected");
    }

    #[test]
    fn test_cdf_large_positive_x() {
        // For a normal distribution (avg = 0, stddev = 1), a very large positive x should have a CDF close to 1
        let x = 5.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cummulative_distrib(x, avg, stddev);
        let expected = 0.999999713; // Approximate value of CDF(5.0)
        assert!((result - expected).abs() < EPSILON, "CDF for x = 5.0 should be very close to 1");
    }

    #[test]
    fn test_cdf_large_negative_x() {
        // For a normal distribution (avg = 0, stddev = 1), a very large negative x should have a CDF close to 0
        let x = -5.0;
        let avg = 0.0;
        let stddev = 1.0;
        let result = cummulative_distrib(x, avg, stddev);
        println!("resuilit large negatif : {:?}", result);
        let expected = 0.000000287; // Approximate value of CDF(-5.0)
        assert!((result - expected).abs() < EPSILON, "CDF for x = -5.0 should be very close to 0");
    }
}
