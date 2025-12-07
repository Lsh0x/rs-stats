/// Provides utility functions for various numerical and combinatorial operations.
pub mod combinatorics;
pub mod constants;
pub mod numeric;

pub use self::combinatorics::{combination, factorial, permutation};
pub use self::constants::{INV_SQRT_2PI, SQRT_2PI, SQRT_2, PI, E, LN_2PI};
pub use self::numeric::{approx_equal, safe_log};
