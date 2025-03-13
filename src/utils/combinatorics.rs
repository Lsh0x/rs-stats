/// Provides functions for combinatorial calculations.

/// Calculate the factorial of a number n.
///
/// # Arguments
/// * `n` - The number to compute the factorial of.
///
/// # Returns
/// * `u64` - The factorial of n.
pub fn factorial(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        _ => (2..=n).fold(1, |acc, x| acc * x),
    }
}

/// Calculate the number of permutations of n items taken k at a time.
///
/// # Arguments
/// * `n` - The total number of items.
/// * `k` - The number of items to choose.
///
/// # Panics
/// If `k` is greater than `n`, this function will panic.
///
/// # Returns
/// * `u64` - The number of permutations.
pub fn permutation(n: u64, k: u64) -> u64 {
    if k > n {
        panic!("k cannot be greater than n");
    }
    ((n - k + 1)..=n).fold(1, |acc, x| acc * x)
}

/// Calculate the number of combinations of n items taken k at a time.
///
/// # Arguments
/// * `n` - The total number of items.
/// * `k` - The number of items to choose.
///
/// # Panics
/// If `k` is greater than `n`, this function will panic.
///
/// # Returns
/// * `u64` - The number of combinations.
pub fn combination(n: u64, k: u64) -> u64 {
    if k > n {
        panic!("k cannot be greater than n");
    }
    let k = if k > n - k { n - k } else { k };
    (1..=k).fold(1, |acc, x| acc * (n - x + 1) / x)
}