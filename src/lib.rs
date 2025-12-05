pub mod distributions;
pub mod error;
pub mod hypothesis_tests;
pub mod prob;
pub mod regression;
pub mod utils;

// Re-export error types for convenience
pub use error::{StatsError, StatsResult};
