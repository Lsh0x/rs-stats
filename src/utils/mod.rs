/// Provides utility functions for various numerical and combinatorial operations.
pub mod combinatorics;
pub mod numeric;

pub use self::combinatorics::{combination, factorial, permutation};
pub use self::numeric::{approx_equal, safe_log};
