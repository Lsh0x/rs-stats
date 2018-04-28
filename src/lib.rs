extern crate num;

/// Mean gets the number expressing the central or typical value in a set of data
pub fn average<'a, T: num::ToPrimitive >(t: &'a [T]) -> Option<f64>  {
    if t.len() == 0 {
        return None
    }
    Some(t.iter().map(|x| x.to_f64().unwrap()).sum::<f64>() / t.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::average;

    #[test]
    fn test_average() {
        assert_eq!(average(&[-2, -3, -4]), Some(-3.0));
        assert_eq!(average(&[1.0, 1.0, 1.0]), Some(1.0));
        assert_eq!(average(&[0, 0, 0]), Some(0.0));
        let mut vec = vec!();
        vec.push(42);
        vec.clear();
        assert_eq!(average(&vec), None);
    }
}
