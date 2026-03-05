//! # rs-stats — Comprehensive Statistical Library
//!
//! `rs-stats` provides a complete, **panic-free** statistical toolkit for Rust:
//! probability functions, 14 parametric distributions with fitting, automatic
//! distribution detection, hypothesis testing, and regression analysis.
//!
//! All fallible operations return [`StatsResult<T>`] — a `Result<T, StatsError>`.
//!
//! ## Modules at a Glance
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`distributions`] | 14 distributions + unified trait interface + auto-fit API |
//! | [`hypothesis_tests`] | t-tests, ANOVA, chi-square, chi-square independence |
//! | [`regression`] | Linear, multiple linear, decision trees |
//! | [`prob`] | Mean, variance, std-dev, z-scores, erf, CDF helpers |
//! | [`utils`] | Special functions (`ln_gamma`, incomplete gamma/beta), combinatorics |
//!
//! ## Medical Quick-Start
//!
//! ### Identify the distribution of a clinical measurement
//!
//! ```rust
//! use rs_stats::{auto_fit, fit_all, Distribution};
//! use rs_stats::distributions::lognormal::LogNormal;
//!
//! // CRP levels (mg/L) from an outpatient cohort — right-skewed biomarker
//! let crp = vec![
//!     0.8, 1.2, 1.5, 2.1, 2.4, 3.2, 3.9, 5.6, 9.7, 12.4,
//!     22.3, 45.0, 88.0, 0.9, 1.3, 1.8, 0.7, 0.6, 0.5, 1.1,
//! ];
//!
//! // One call: auto-detect type, fit all candidates, return the best (lowest AIC)
//! let best = auto_fit(&crp).unwrap();
//! println!("Best distribution: {} (AIC={:.2})", best.name, best.aic);
//! // → Best distribution: LogNormal
//!
//! // Fit explicitly and answer clinical questions
//! let crp_dist = LogNormal::fit(&crp).unwrap();
//! let median   = crp_dist.inverse_cdf(0.5).unwrap();
//! let p_high   = 1.0 - crp_dist.cdf(10.0).unwrap();  // P(CRP > 10 mg/L)
//! println!("Median CRP        = {:.2} mg/L", median);
//! println!("P(CRP > 10 mg/L)  = {:.1}%", p_high * 100.0);
//! ```
//!
//! ### Compare two treatment arms
//!
//! ```rust
//! use rs_stats::hypothesis_tests::t_test::two_sample_t_test;
//!
//! // Systolic blood pressure (mmHg) — control vs. treated
//! let control   = vec![128.0, 132.0, 125.0, 130.0, 129.0, 131.0];
//! let treatment = vec![118.0, 122.0, 115.0, 120.0, 119.0, 121.0];
//!
//! let result = two_sample_t_test(&control, &treatment, false).unwrap();
//! println!("t = {:.3}, p = {:.4}", result.t_statistic, result.p_value);
//! if result.p_value < 0.05 {
//!     println!("→ Significant BP reduction (α = 0.05)");
//! }
//! ```
//!
//! ### Compute reference intervals from data
//!
//! ```rust
//! use rs_stats::distributions::normal_distribution::Normal;
//! use rs_stats::Distribution;
//!
//! // Haemoglobin (g/dL) in healthy adults — estimate the 95% reference interval
//! let hgb = vec![13.5, 14.2, 13.8, 15.1, 14.5, 13.9, 14.8, 15.3, 14.0, 14.6];
//! let dist = Normal::fit(&hgb).unwrap();
//!
//! let lower = dist.inverse_cdf(0.025).unwrap();
//! let upper = dist.inverse_cdf(0.975).unwrap();
//! println!("95% reference interval: [{:.2}, {:.2}] g/dL", lower, upper);
//! ```
//!
//! ## Trait-Based Polymorphism
//!
//! Use `Box<dyn Distribution>` to work with distributions at runtime:
//!
//! ```rust
//! use rs_stats::Distribution;
//! use rs_stats::distributions::{
//!     normal_distribution::Normal,
//!     lognormal::LogNormal,
//! };
//!
//! // Choose the distribution based on data characteristics
//! fn best_model(skewed: bool) -> Box<dyn Distribution> {
//!     if skewed {
//!         Box::new(LogNormal::new(1.0, 0.5).unwrap())
//!     } else {
//!         Box::new(Normal::new(80.0, 10.0).unwrap())
//!     }
//! }
//!
//! let model = best_model(true);
//! println!("Mean = {:.2}", model.mean());
//! println!("P95  = {:.2}", model.inverse_cdf(0.95).unwrap());
//! ```
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
    DataKind, FitResult, KsResult, SkippedFit, auto_fit, detect_data_type, fit_all,
    fit_all_discrete, fit_all_discrete_verbose, fit_all_verbose, fit_best, fit_best_discrete,
};
// Re-export traits for use without the full path
pub use distributions::traits::{DiscreteDistribution, Distribution};
