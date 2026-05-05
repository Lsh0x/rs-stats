//! Basic probability primitives — descriptive statistics, error function,
//! Welford online estimators.
//!
//! Distribution-specific PDF/CDF/quantile functions live on the per-distribution
//! types in [`crate::distributions`]; for the standard normal CDF/PDF use
//! [`crate::distributions::normal_distribution::Normal`] directly.

pub mod average;
pub mod erf;
pub mod erfc;
pub mod std_dev;
pub mod std_err;
pub mod variance;
pub mod welford;
pub mod z_score;

pub use self::average::average;
pub use self::erf::erf;
pub use self::erfc::erfc;
pub use self::std_dev::{std_dev, std_dev_population, std_dev_sample};
pub use self::std_err::std_err;
pub use self::variance::{variance, variance_population, variance_sample};
pub use self::welford::{Welford, WelfordCovariance, WelfordVector};
pub use self::z_score::z_score;
