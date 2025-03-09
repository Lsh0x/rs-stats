/// Provides numerical utility functions for statistical calculations.

/// Computes the natural logarithm of x, handling edge cases safely.
/// 
/// # Arguments
/// * `x` - The input number.
/// 
/// # Returns
/// * `f64` - The natural logarithm of x.
/// 
/// # Panics
/// If x is less than or equal to 0, this function will panic.
pub fn safe_log(x: f64) -> f64 {
    if x <= 0.0 {
        panic!("Logarithm is only defined for positive numbers.");
    }
    x.ln()
}

/// Checks if two floating-point numbers are approximately equal within a given epsilon.
/// 
/// # Arguments
/// * `a` - The first number.
/// * `b` - The second number.
/// * `epsilon` - The maximum allowed difference between a and b.
/// 
/// # Returns
/// * `bool` - True if the numbers are approximately equal, False otherwise.
pub fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() <= epsilon
}