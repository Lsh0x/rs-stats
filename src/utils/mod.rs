/// Provides utility functions for various numerical and combinatorial operations.
pub mod combinatorics;
pub mod constants;
pub mod numeric;
pub mod special_functions;

pub use self::combinatorics::{combination, factorial, permutation};
pub use self::constants::{E, INV_SQRT_2PI, LN_2PI, PI, SQRT_2, SQRT_2PI};
pub use self::numeric::{approx_equal, safe_log};
pub use self::special_functions::{
    beta_fn, bisect_inverse_cdf, gamma_fn, ln_beta, ln_gamma, regularized_incomplete_beta,
    regularized_incomplete_gamma,
};
