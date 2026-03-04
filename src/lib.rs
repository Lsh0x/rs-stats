pub mod distributions;
pub mod error;
pub mod hypothesis_tests;
pub mod prob;
pub mod regression;
pub mod utils;

// Re-export error types for convenience
pub use error::{StatsError, StatsResult};

// Re-export fitting API at the crate root for easy access
pub use distributions::fitting::{
    DataKind, FitResult, KsResult, auto_fit, detect_data_type, fit_all, fit_all_discrete, fit_best,
    fit_best_discrete,
};
// Re-export traits for use without the full path
pub use distributions::traits::{DiscreteDistribution, Distribution};
