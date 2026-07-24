// ── Existing distributions ─────────────────────────────────────────────────────
pub mod binomial_distribution;
pub mod exponential_distribution;
pub mod normal_distribution;
pub mod poisson_distribution;
pub mod uniform_distribution;

// ── New continuous distributions ───────────────────────────────────────────────
pub mod beta;
pub mod cauchy;
pub mod chi_squared;
pub mod f_distribution;
pub mod gamma_distribution;
pub mod laplace;
pub mod logistic;
pub mod lognormal;
pub mod pareto;
pub mod student_t;
pub mod weibull;

// ── New discrete distributions ─────────────────────────────────────────────────
pub mod geometric;
pub mod negative_binomial;

// ── Traits & fitting ───────────────────────────────────────────────────────────
pub mod multivariate_normal;

pub mod fitting;
pub mod traits;

// ── Flat re-exports for ergonomic imports ──────────────────────────────────────
// Allows `use rs_stats::distributions::Weibull` instead of the full module path.
// Continuous — existing
pub use binomial_distribution::Binomial;
pub use exponential_distribution::Exponential;
pub use multivariate_normal::MultivariateNormal;
pub use normal_distribution::Normal;
pub use poisson_distribution::Poisson;
pub use uniform_distribution::Uniform;
// Continuous — new
pub use beta::Beta;
pub use cauchy::Cauchy;
pub use chi_squared::ChiSquared;
pub use f_distribution::FDistribution;
pub use gamma_distribution::Gamma;
pub use laplace::Laplace;
pub use logistic::Logistic;
pub use lognormal::LogNormal;
pub use pareto::Pareto;
pub use student_t::StudentT;
pub use weibull::Weibull;
// Discrete — new
pub use geometric::Geometric;
pub use negative_binomial::NegativeBinomial;
