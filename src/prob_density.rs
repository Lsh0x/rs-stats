use crate::z_score;

use std::f64::consts::PI;

/// probability_density normalize x using the mean and the standard deviation and return the PDF
/// https://en.wikipedia.org/wiki/Probability_density_function
#[inline]
pub fn probability_density(x: f64, avg: f64, stddev: f64) -> f64 {
    (z_score(x, avg, stddev).powi(2) / -2.0).exp() / (stddev * (PI * 2.0).sqrt())
}

/// normal_probability_density return the PDF with z already normalized
/// https://en.wikipedia.org/wiki/Probability_density_function
#[inline]
pub fn normal_probability_density(z: f64) -> f64 {
    (z.powi(2) / -2.0).exp() / (PI * 2.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_density_basic() {
        let avg = 0.0;
        let stddev = 1.0;

        let test_cases = vec![
            (0.0, 0.3989422804014327),   // Peak of the distribution
            (1.0, 0.24197072451914337),  // One standard deviation away
            (-1.0, 0.24197072451914337), // One standard deviation away (symmetry)
            (2.0, 0.05399096651318806),  // Two standard deviations away
            (3.0, 0.00443184841193801),  // Three standard deviations away
        ];

        for (x, expected) in test_cases {
            let actual = probability_density(x, avg, stddev);
            assert!(
                (actual - expected).abs() < 1e-10,
                "For x = {}, expected {}, but got {}",
                x,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_probability_density_different_mean() {
        let avg = 5.0;
        let stddev = 2.0;
        let x = 7.0; // One standard deviation above the mean
        let expected = 0.12098536225957168;
        let actual = probability_density(x, avg, stddev);
        assert!(
            (actual - expected).abs() < 1e-10,
            "For x = {}, expected {}, but got {}",
            x,
            expected,
            actual
        );
    }

    #[test]
    fn test_probability_density_different_stddev() {
        let avg = 0.0;
        let stddev = 0.5;
        let x = 0.0;
        let expected = 0.7978845608028654;
        let actual = probability_density(x, avg, stddev);
        assert!(
            (actual - expected).abs() < 1e-10,
            "For x = {}, expected {}, but got {}",
            x,
            expected,
            actual
        );
    }

    #[test]
    fn test_normal_probability_density_basic() {
        let test_cases = vec![
            (0.0, 0.3989422804014327),   // Peak of the distribution
            (1.0, 0.24197072451914337),  // One standard deviation away
            (-1.0, 0.24197072451914337), // One standard deviation away (symmetry)
            (2.0, 0.05399096651318806),  // Two standard deviations away
            (3.0, 0.00443184841193801),  // Three standard deviations away
        ];

        for (z, expected) in test_cases {
            let actual = normal_probability_density(z);
            assert!(
                (actual - expected).abs() < 1e-10,
                "For z = {}, expected {}, but got {}",
                z,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_normal_probability_density_symmetry() {
        let z = 0.7;
        let actual = normal_probability_density(z) - normal_probability_density(-z);
        assert!(
            actual.abs() < 1e-10,
            "normal_probability_density(z) should equal normal_probability_density(-z), but got {}",
            actual
        );
    }

    #[test]
    fn test_normal_probability_density_limits() {
        // Test approaching limits
        assert!(normal_probability_density(10.0) < 1e-20); // PDF -> 0 as z -> +/-inf
        assert!(normal_probability_density(-10.0) < 1e-20);
    }
}
