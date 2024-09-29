/// the zscore represente the distance from the mean in stddev
#[inline]
pub fn z_score(x: f64, avg: f64, stddev: f64) -> f64 {
    if stddev == 0.0 {
        return f64::INFINITY;
    }
    (x - avg) / stddev
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_z_score_integer() {
        let x = 5.0;
        let avg = 3.0;
        let stddev = 2.0;
        let result = z_score(x, avg, stddev);
        let expected = (5.0 - 3.0) / 2.0;  // (x - avg) / stddev
        assert!((result - expected).abs() < EPSILON, "Z-score for value 5 with avg 3 and stddev 2 should match expected");
    }

    #[test]
    fn test_z_score_float() {
        let x = 4.5;
        let avg = 3.0;
        let stddev = 1.5;
        let result = z_score(x, avg, stddev);
        let expected = (4.5 - 3.0) / 1.5;  // (x - avg) / stddev
        assert!((result - expected).abs() < EPSILON, "Z-score for value 4.5 with avg 3 and stddev 1.5 should match expected");
    }

    #[test]
    fn test_z_score_negative() {
        let x = 1.0;
        let avg = 3.0;
        let stddev = 2.0;
        let result = z_score(x, avg, stddev);
        let expected = (1.0 - 3.0) / 2.0;  // (x - avg) / stddev
        assert!((result - expected).abs() < EPSILON, "Z-score for value 1 with avg 3 and stddev 2 should match expected");
    }

    #[test]
    fn test_z_score_zero_stddev() {
        let x = 3.0;
        let avg = 3.0;
        let stddev = 0.0;
        let result = z_score(x, avg, stddev);
        assert!(result.is_infinite(), "Z-score should be infinite when stddev is 0");
    }

    #[test]
    fn test_z_score_zero_mean() {
        let x = 3.0;
        let avg = 0.0;
        let stddev = 2.0;
        let result = z_score(x, avg, stddev);
        let expected = (3.0 - 0.0) / 2.0;
        assert!((result - expected).abs() < EPSILON, "Z-score for value 3 with avg 0 and stddev 2 should match expected");
    }
}