// ── Existing distributions ─────────────────────────────────────────────────────
pub mod binomial_distribution;
pub mod exponential_distribution;
pub mod normal_distribution;
pub mod poisson_distribution;
pub mod uniform_distribution;

// ── New continuous distributions ───────────────────────────────────────────────
pub mod beta;
pub mod chi_squared;
pub mod f_distribution;
pub mod gamma_distribution;
pub mod lognormal;
pub mod student_t;
pub mod weibull;

// ── New discrete distributions ─────────────────────────────────────────────────
pub mod geometric;
pub mod negative_binomial;

// ── Traits & fitting ───────────────────────────────────────────────────────────
pub mod fitting;
pub mod traits;

// ── Flat re-exports for ergonomic imports ──────────────────────────────────────
// Allows `use rs_stats::distributions::Weibull` instead of the full module path.
// Continuous — existing
pub use binomial_distribution::Binomial;
pub use exponential_distribution::Exponential;
pub use normal_distribution::Normal;
pub use poisson_distribution::Poisson;
pub use uniform_distribution::Uniform;
// Continuous — new
pub use beta::Beta;
pub use chi_squared::ChiSquared;
pub use f_distribution::FDistribution;
pub use gamma_distribution::Gamma;
pub use lognormal::LogNormal;
pub use student_t::StudentT;
pub use weibull::Weibull;
// Discrete — new
pub use geometric::Geometric;
pub use negative_binomial::NegativeBinomial;
