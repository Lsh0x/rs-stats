use crate::variance;

/// std_dev return the standard deviation, the square root of the variance
#[inline]
pub fn std_dev<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    variance(t).map(|x| x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_population_std_dev_integers() {
        let data = vec![1, 2, 3, 4, 5];
        let result = std_dev(&data);
        let expected = 2.0_f64.sqrt(); // sqrt of population variance (2.0)
        assert!(
            (result.unwrap() - expected).abs() < EPSILON,
            "Population std_dev for integers should be sqrt(2.0)"
        );
    }

    #[test]
    fn test_population_std_dev_floats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_dev(&data);
        let expected = 2.0_f64.sqrt(); // sqrt of population variance (2.0)
        assert!(
            (result.unwrap() - expected).abs() < EPSILON,
            "Population std_dev for floats should be sqrt(2.0)"
        );
    }

    #[test]
    fn test_population_std_dev_mixed_floats() {
        let data = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = std_dev(&data);
        let expected = 2.0_f64.sqrt(); // sqrt of population variance (2.0)
        assert!(
            (result.unwrap() - expected).abs() < EPSILON,
            "Population std_dev for mixed floats should be sqrt(2.0)"
        );
    }
}
