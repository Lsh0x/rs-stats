/// Provides utility functions for various numerical and combinatorial operations.
pub mod combinatorics;
pub mod constants;
pub mod numeric;

pub use self::combinatorics::{combination, factorial, permutation};
pub use self::constants::{E, INV_SQRT_2PI, LN_2PI, PI, SQRT_2, SQRT_2PI};
pub use self::numeric::{approx_equal, safe_log};
