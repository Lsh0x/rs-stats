/// average gets the number expressing the central or typical value in a set of data
#[inline]
pub fn average<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    if t.is_empty() {
        return None;
    }
    Some(t.iter().map(|x| x.to_f64().unwrap()).sum::<f64>() / t.len() as f64)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_integers() {
        let data = vec![1, 2, 3, 4, 5];
        let result = average(&data);
        assert_eq!(result, Some(3.0));
    }

    #[test]
    fn test_average_floats() {
        let data = vec![1.5, 2.5, 3.5, 4.5];
        let result = average(&data);
        assert_eq!(result, Some(3.0));
    }

    #[test]
    fn test_average_mixed_types() {
        let data = vec![1.0, 2.0, 3.0, 4.5, 5.5]; // All elements are f64
        let result = average(&data);
        assert_eq!(result, Some(3.2));
    }

    #[test]
    fn test_average_empty_slice() {
        let data: Vec<f64> = Vec::new();
        let result = average(&data);
        assert_eq!(result, None);
    }

    #[test]
    fn test_average_single_value() {
        let data = vec![10.0];
        let result = average(&data);
        assert_eq!(result, Some(10.0));
    }
}
