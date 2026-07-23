//! Provides functions for combinatorial calculations.

use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::ln_gamma;

/// Calculate the factorial of a number n.
///
/// # Arguments
/// * `n` - The number to compute the factorial of.
///
/// # Returns
/// * `StatsResult<u64>` - The factorial of n, or an error if n >= 21 (overflow).
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `n >= 21` (result overflows `u64`).
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::factorial;
///
/// assert_eq!(factorial(0).unwrap(), 1);
/// assert_eq!(factorial(5).unwrap(), 120);
/// assert!(factorial(21).is_err()); // Overflows u64
/// ```
pub fn factorial(n: u64) -> StatsResult<u64> {
    match n {
        0 | 1 => Ok(1),
        _ => {
            let mut result: u64 = 1;
            for i in 2..=n {
                result = result.checked_mul(i).ok_or_else(|| {
                    StatsError::invalid_input(format!(
                        "factorial({}) overflows u64 (max supported: factorial(20))",
                        n
                    ))
                })?;
            }
            Ok(result)
        }
    }
}

/// Calculate the number of permutations of n items taken k at a time.
///
/// # Arguments
/// * `n` - The total number of items.
/// * `k` - The number of items to choose.
///
/// # Returns
/// * `StatsResult<u64>` - The number of permutations, or an error if k > n
///   or the result overflows `u64`.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k > n`, or if the result overflows
/// `u64` (use [`ln_permutation`] for large arguments).
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::permutation;
///
/// let result = permutation(5, 3).unwrap();
/// assert_eq!(result, 60);
///
/// // Error cases
/// assert!(permutation(5, 10).is_err());
/// assert!(permutation(30, 20).is_err()); // overflows u64
/// ```
pub fn permutation(n: u64, k: u64) -> StatsResult<u64> {
    if k > n {
        return Err(StatsError::invalid_input(format!(
            "k ({}) cannot be greater than n ({})",
            k, n
        )));
    }
    let mut result: u64 = 1;
    for i in (n - k + 1)..=n {
        result = result.checked_mul(i).ok_or_else(|| {
            StatsError::invalid_input(format!(
                "permutation({n}, {k}) overflows u64; use ln_permutation for large arguments"
            ))
        })?;
    }
    Ok(result)
}

/// Natural logarithm of the number of permutations:
/// `ln P(n, k) = ln Γ(n+1) − ln Γ(n−k+1)`.
///
/// Never overflows — use this when [`permutation`] exceeds `u64`.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k > n`.
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::{ln_permutation, permutation};
///
/// let exact = permutation(10, 3).unwrap() as f64;
/// let ln = ln_permutation(10, 3).unwrap();
/// assert!((ln - exact.ln()).abs() < 1e-10);
/// ```
pub fn ln_permutation(n: u64, k: u64) -> StatsResult<f64> {
    if k > n {
        return Err(StatsError::invalid_input(format!(
            "k ({}) cannot be greater than n ({})",
            k, n
        )));
    }
    Ok(ln_gamma(n as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0))
}

/// Calculate the number of combinations of n items taken k at a time.
///
/// # Arguments
/// * `n` - The total number of items.
/// * `k` - The number of items to choose.
///
/// # Returns
/// * `StatsResult<u64>` - The number of combinations, or an error if k > n
///   or the result overflows `u64`.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k > n`, or if the result overflows
/// `u64` (use [`ln_combination`] for large arguments).
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::combination;
///
/// let result = combination(5, 3).unwrap();
/// assert_eq!(result, 10);
///
/// // C(66, 33) fits in u64 even though naive intermediates overflow.
/// assert_eq!(combination(66, 33).unwrap(), 7_219_428_434_016_265_740);
///
/// // Error cases
/// assert!(combination(5, 10).is_err());
/// assert!(combination(70, 35).is_err()); // overflows u64
/// ```
pub fn combination(n: u64, k: u64) -> StatsResult<u64> {
    if k > n {
        return Err(StatsError::invalid_input(format!(
            "k ({}) cannot be greater than n ({})",
            k, n
        )));
    }
    let k = if k > n - k { n - k } else { k };
    // Intermediates are held in u128: after step x the accumulator equals
    // C(n, x), but the product before the division is C(n, x)·x, which can
    // overflow u64 even when the final C(n, k) fits (e.g. C(66, 33)).
    let overflow = || {
        StatsError::invalid_input(format!(
            "combination({n}, {k}) overflows u64; use ln_combination for large arguments"
        ))
    };
    let mut acc: u128 = 1;
    for x in 1..=k {
        acc = acc.checked_mul((n - x + 1) as u128).ok_or_else(overflow)? / (x as u128);
    }
    u64::try_from(acc).map_err(|_| overflow())
}

/// Natural logarithm of the number of combinations:
/// `ln C(n, k) = ln Γ(n+1) − ln Γ(k+1) − ln Γ(n−k+1)`.
///
/// Never overflows — use this when [`combination`] exceeds `u64`.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k > n`.
///
/// # Examples
/// ```
/// use rs_stats::utils::combinatorics::{combination, ln_combination};
///
/// let exact = combination(10, 3).unwrap() as f64;
/// let ln = ln_combination(10, 3).unwrap();
/// assert!((ln - exact.ln()).abs() < 1e-10);
/// ```
pub fn ln_combination(n: u64, k: u64) -> StatsResult<f64> {
    if k > n {
        return Err(StatsError::invalid_input(format!(
            "k ({}) cannot be greater than n ({})",
            k, n
        )));
    }
    Ok(ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0).unwrap(), 1);
        assert_eq!(factorial(1).unwrap(), 1);
        assert_eq!(factorial(5).unwrap(), 120);
        assert_eq!(factorial(10).unwrap(), 3628800);
        assert_eq!(factorial(20).unwrap(), 2_432_902_008_176_640_000);
    }

    #[test]
    fn test_factorial_overflow() {
        // factorial(21) overflows u64
        assert!(factorial(21).is_err());
        assert!(matches!(
            factorial(21).unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
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
    fn test_permutation_overflow() {
        // P(30, 20) = 30!/10! ≈ 7.3e25 > u64::MAX — must error, never wrap.
        assert!(permutation(30, 20).is_err());
        // P(20, 10) fits.
        assert_eq!(permutation(20, 10).unwrap(), 670_442_572_800);
    }

    #[test]
    fn test_combination_u64_edge() {
        // Fits in u64 but naive u64 intermediates overflow.
        assert_eq!(combination(66, 33).unwrap(), 7_219_428_434_016_265_740);
        assert_eq!(combination(62, 31).unwrap(), 465_428_353_255_261_088);
        // Result itself exceeds u64 — typed error.
        assert!(combination(70, 35).is_err());
    }

    #[test]
    fn test_ln_variants() {
        // Agree with exact values where those exist…
        let exact = combination(52, 5).unwrap() as f64;
        assert!((ln_combination(52, 5).unwrap() - exact.ln()).abs() < 1e-9);
        let exact = permutation(20, 10).unwrap() as f64;
        assert!((ln_permutation(20, 10).unwrap() - exact.ln()).abs() < 1e-9);
        // …and stay finite far beyond u64 range.
        assert!(ln_combination(1000, 500).unwrap().is_finite());
        assert!(ln_permutation(1000, 500).unwrap().is_finite());
        // Same domain validation as the exact versions.
        assert!(ln_combination(5, 10).is_err());
        assert!(ln_permutation(5, 10).is_err());
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

        assert_eq!(
            result1, result2,
            "C(n, k) should equal C(n, n-k) when k > n-k"
        );
        assert_eq!(result1, 45u64, "C(10, 8) should equal C(10, 2) = 45");
    }
}
