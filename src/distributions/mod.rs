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
