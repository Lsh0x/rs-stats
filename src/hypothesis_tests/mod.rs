pub mod t_test;
pub mod chi_square_test;
pub mod anova;

// Re-export functions to allow users to import them directly from hypothesis_tests module
pub use self::t_test::{one_sample_t_test, two_sample_t_test, paired_t_test};
pub use self::chi_square_test::{chi_square_goodness_of_fit, chi_square_independence};
pub use self::anova::one_way_anova;

