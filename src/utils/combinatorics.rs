//! Provides functions for combinatorial calculations.

use crate::error::{StatsError, StatsResult};

/// Calculate the factorial of a number n.
///
/// # Arguments
/// * `n` - The number to compute the factorial of.
///
/// # Returns
/// * `u64` - The factorial of n.
///
/// # Note
/// This function does not return a Result because factorial is always valid
/// for any u64 input (though it may overflow for large values).
pub fn factorial(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        _ => (2..=n).product::<u64>(),
    }
}

/// Calculate the number of permutations of n items taken k at a time.
///
/// # Arguments
/// * `n` - The total number of items.
/// * `k` - The number of items to choose.
///
/// # Returns
/// * `StatsResult<u64>` - The number of permutations, or an error if k > n.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k > n`.
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::permutation;
///
/// let result = permutation(5, 3).unwrap();
/// assert_eq!(result, 60);
///
/// // Error case
/// assert!(permutation(5, 10).is_err());
/// ```
pub fn permutation(n: u64, k: u64) -> StatsResult<u64> {
    if k > n {
        return Err(StatsError::invalid_input(format!(
            "k ({}) cannot be greater than n ({})",
            k, n
        )));
    }
    Ok(((n - k + 1)..=n).product::<u64>())
}

/// Calculate the number of combinations of n items taken k at a time.
///
/// # Arguments
/// * `n` - The total number of items.
/// * `k` - The number of items to choose.
///
/// # Returns
/// * `StatsResult<u64>` - The number of combinations, or an error if k > n.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k > n`.
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::combination;
///
/// let result = combination(5, 3).unwrap();
/// assert_eq!(result, 10);
///
/// // Error case
/// assert!(combination(5, 10).is_err());
/// ```
pub fn combination(n: u64, k: u64) -> StatsResult<u64> {
    if k > n {
        return Err(StatsError::invalid_input(format!(
            "k ({}) cannot be greater than n ({})",
            k, n
        )));
    }
    let k = if k > n - k { n - k } else { k };
    Ok((1..=k).fold(1, |acc, x| acc * (n - x + 1) / x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_permutation_valid() {
        assert_eq!(permutation(5, 3).unwrap(), 60);
        assert_eq!(permutation(5, 5).unwrap(), 120);
        assert_eq!(permutation(5, 0).unwrap(), 1);
        assert_eq!(permutation(10, 3).unwrap(), 720);
    }

    #[test]
    fn test_permutation_invalid() {
        assert!(permutation(5, 10).is_err());
        assert!(matches!(
            permutation(5, 10).unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_combination_valid() {
        assert_eq!(combination(5, 3).unwrap(), 10);
        assert_eq!(combination(5, 5).unwrap(), 1);
        assert_eq!(combination(5, 0).unwrap(), 1);
        assert_eq!(combination(10, 3).unwrap(), 120);
    }

    #[test]
    fn test_combination_invalid() {
        assert!(combination(5, 10).is_err());
        assert!(matches!(
            combination(5, 10).unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_combination_symmetry() {
        // C(n, k) = C(n, n-k)
        assert_eq!(combination(10, 3).unwrap(), combination(10, 7).unwrap());
        assert_eq!(combination(20, 5).unwrap(), combination(20, 15).unwrap());
    }

    #[test]
    fn test_combination_k_greater_than_n_minus_k() {
        // Test the symmetry optimization path when k > n - k
        // This tests the internal optimization in combination()
        let n = 10u64;
        let k = 8u64; // k > n - k (8 > 2)
        
        // This should use the symmetry path: combination(10, 8) = combination(10, 2)
        let result1 = combination(n, k).unwrap();
        let result2 = combination(n, n - k).unwrap();
        
        assert_eq!(result1, result2, "C(n, k) should equal C(n, n-k) when k > n-k");
        assert_eq!(result1, 45u64, "C(10, 8) should equal C(10, 2) = 45");
    }
}
