pub mod anova;
pub mod chi_square_test;
pub mod fisher;
pub mod ks;
pub mod mann_whitney;
pub mod normality;
pub mod p_adjust;
pub mod power;
pub mod t_test;
pub mod wilcoxon;

// Re-export functions to allow users to import them directly from hypothesis_tests module
pub use self::anova::one_way_anova;
pub use self::chi_square_test::{
    ChiSquareResult, chi_square_goodness_of_fit, chi_square_independence,
};
pub use self::fisher::{FisherExactResult, fisher_exact};
pub use self::ks::ks_2samp;
pub use self::mann_whitney::{MannWhitneyResult, mann_whitney_u};
pub use self::normality::{NormalityResult, dagostino_k2};
pub use self::p_adjust::{PAdjustMethod, p_adjust};
pub use self::power::{TTestKind, power_t_test, sample_size_t_test};
pub use self::t_test::{
    cohens_d, one_sample_t_test, one_sample_t_test_alt, paired_t_test, paired_t_test_alt,
    two_sample_t_test, two_sample_t_test_alt,
};
pub use self::wilcoxon::{WilcoxonResult, wilcoxon_signed_rank};

/// Alternative hypothesis for one- and two-sided tests.
///
/// Follows the scipy convention: for a two-sample statistic on `(a, b)`,
/// `Less` means "the first sample is stochastically smaller".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alternative {
    /// H₁: the parameter differs from the null value (default).
    #[default]
    TwoSided,
    /// H₁: the parameter is smaller than the null value.
    Less,
    /// H₁: the parameter is greater than the null value.
    Greater,
}
