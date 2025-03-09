pub mod average;
pub mod cumulative_distrib;
pub mod erf;
pub mod erfc;
pub mod normal_cumulative_distrib;
pub mod prob_density;
pub mod std_dev;
pub mod std_err;
pub mod variance;
pub mod z_score;

// Re-export functions to allow users to import them directly from prob module
pub use self::average::average;
pub use self::cumulative_distrib::cumulative_distrib;
pub use self::erf::erf;
pub use self::erfc::erfc;
pub use self::normal_cumulative_distrib::normal_cumulative_distrib;
pub use self::prob_density::probability_density;
pub use self::std_dev::std_dev;
pub use self::std_err::std_err;
pub use self::variance::variance;
pub use self::z_score::z_score;
