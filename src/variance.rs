use crate::average;

/// variance cs the mean of the sum of all square deviation
pub fn variance<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    match average(t) {
        Some(avg) => {
            let len: f64 = t.len() as f64;
            Some(
                t.iter()
                    .map(|x| (x.to_f64().unwrap() - avg).powi(2))
                    .sum::<f64>()
                    / len,
            )
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_integers() {
        let data = vec![1, 2, 3, 4, 5];
        let result = variance(&data);
        assert_eq!(result, Some(2.0)); // Variance of [1, 2, 3, 4, 5] is 2.0
    }

    #[test]
    fn test_variance_floats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = variance(&data);
        assert_eq!(result, Some(2.0)); // Variance of [1.0, 2.0, 3.0, 4.0, 5.0] is 2.0
    }

    #[test]
    fn test_variance_mixed_floats() {
        let data = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = variance(&data);
        assert_eq!(result, Some(2.0)); // Variance of [1.5, 2.5, 3.5, 4.5, 5.5] is 2.0
    }

    #[test]
    fn test_variance_single_value() {
        let data = vec![10];
        let result = variance(&data);
        assert_eq!(result, Some(0.0)); // Variance of [10] is 0.0
    }

    #[test]
    fn test_variance_empty_slice() {
        let data: Vec<f64> = Vec::new(); // Empty vector
        let result = variance(&data);
        assert_eq!(result, None); // Variance of an empty list is None
    }

    #[test]
    fn test_variance_identical_values() {
        let data = vec![4.0, 4.0, 4.0, 4.0]; // All elements are the same
        let result = variance(&data);
        assert_eq!(result, Some(0.0)); // Variance is 0.0 because there's no variation
    }
}
