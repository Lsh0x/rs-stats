//! # Uniform Distribution
//!
//! This module implements the Uniform distribution, a continuous probability distribution
//! where all values within a given range are equally likely to occur.
//!
//! ## Key Characteristics
//! - All values in the range [a, b] have equal probability
//! - Constant PDF within the range, zero outside
//! - Linear CDF function from 0 to 1 over the range
//!
//! ## Common Applications
//! - Modeling random variables with no bias toward any value in a range
//! - Generating random numbers for Monte Carlo simulations
//! - Baseline distribution for comparing other distributions
//!
//! ## Mathematical Formulation
//! For a uniform distribution over the interval [a, b]:
//!
//! PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
//! CDF: F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
//! Mean: (a + b)/2
//! Variance: (b - a)²/12

use crate::error::{StatsError, StatsResult};
// Private math helpers; the public API is the [`Uniform`] struct's
// [`Distribution`] impl below.

#[inline]
fn uniform_pdf(x: f64, a: f64, b: f64) -> StatsResult<f64> {
    if a >= b {
        return Err(StatsError::InvalidInput {
            message: "uniform_pdf: a must be less than b".to_string(),
        });
    }
    Ok(if x < a || x > b { 0.0 } else { 1.0 / (b - a) })
}

#[inline]
fn uniform_cdf(x: f64, a: f64, b: f64) -> StatsResult<f64> {
    if a >= b {
        return Err(StatsError::InvalidInput {
            message: "uniform_cdf: a must be less than b".to_string(),
        });
    }
    Ok(if x < a {
        0.0
    } else if x > b {
        1.0
    } else {
        (x - a) / (b - a)
    })
}

#[inline]
fn uniform_inverse_cdf(p: f64, a: f64, b: f64) -> StatsResult<f64> {
    if a >= b {
        return Err(StatsError::InvalidInput {
            message: "uniform_inverse_cdf: a must be less than b".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::InvalidInput {
            message: "uniform_inverse_cdf: p must be between 0 and 1".to_string(),
        });
    }
    Ok(a + p * (b - a))
}

// ── Typed struct + Distribution impl ──────────────────────────────────────────

/// Uniform distribution U(a, b) as a typed struct.
///
/// # Examples
/// ```
/// use rs_stats::distributions::uniform_distribution::Uniform;
/// use rs_stats::distributions::traits::Distribution;
///
/// let u = Uniform::new(0.0, 1.0).unwrap();
/// assert!((u.mean() - 0.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    /// Lower bound a
    pub a: f64,
    /// Upper bound b (must be > a)
    pub b: f64,
}

impl Uniform {
    /// Creates a `Uniform` distribution with validation (requires a < b).
    pub fn new(a: f64, b: f64) -> StatsResult<Self> {
        if a >= b {
            return Err(StatsError::InvalidInput {
                message: "Uniform::new: a must be strictly less than b".to_string(),
            });
        }
        Ok(Self { a, b })
    }

    /// MLE: a = min(data), b = max(data).
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Uniform::fit: data must not be empty".to_string(),
            });
        }
        let a = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let b = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self::new(a, b)
    }
}

impl crate::distributions::traits::Distribution for Uniform {
    fn name(&self) -> &str {
        "Uniform"
    }
    fn num_params(&self) -> usize {
        2
    }
    fn pdf(&self, x: f64) -> StatsResult<f64> {
        uniform_pdf(x, self.a, self.b)
    }
    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        if x < self.a || x > self.b {
            // log(0) = -∞; return a large negative number
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(-((self.b - self.a).ln()))
        }
    }
    fn cdf(&self, x: f64) -> StatsResult<f64> {
        uniform_cdf(x, self.a, self.b)
    }
    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        uniform_inverse_cdf(p, self.a, self.b)
    }
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_uniform_pdf_inside_range() {
        // For a uniform distribution on [0, 1], PDF should be 1 inside the range
        assert!((uniform_pdf(0.0, 0.0, 1.0).unwrap() - 1.0).abs() < EPSILON);
        assert!((uniform_pdf(0.5, 0.0, 1.0).unwrap() - 1.0).abs() < EPSILON);
        assert!((uniform_pdf(1.0, 0.0, 1.0).unwrap() - 1.0).abs() < EPSILON);

        // For a uniform distribution on [2, 4], PDF should be 1/2 inside the range
        assert!((uniform_pdf(2.0, 2.0, 4.0).unwrap() - 0.5).abs() < EPSILON);
        assert!((uniform_pdf(3.0, 2.0, 4.0).unwrap() - 0.5).abs() < EPSILON);
        assert!((uniform_pdf(4.0, 2.0, 4.0).unwrap() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_pdf_outside_range() {
        // PDF should be 0 outside the range
        assert!((uniform_pdf(-1.0, 0.0, 1.0).unwrap() - 0.0).abs() < EPSILON);
        assert!((uniform_pdf(2.0, 0.0, 1.0).unwrap() - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_pdf_invalid_range() {
        // Should panic if a >= b
        let result = uniform_pdf(0.5, 1.0, 0.0);
        assert!(result.is_err(), "Should return error for a >= b");
    }

    #[test]
    fn test_uniform_cdf_inside_range() {
        // For a uniform distribution on [0, 1], CDF should increase linearly from 0 to 1
        assert!((uniform_cdf(0.0, 0.0, 1.0).unwrap() - 0.0).abs() < EPSILON);
        assert!((uniform_cdf(0.25, 0.0, 1.0).unwrap() - 0.25).abs() < EPSILON);
        assert!((uniform_cdf(0.5, 0.0, 1.0).unwrap() - 0.5).abs() < EPSILON);
        assert!((uniform_cdf(0.75, 0.0, 1.0).unwrap() - 0.75).abs() < EPSILON);
        assert!((uniform_cdf(1.0, 0.0, 1.0).unwrap() - 1.0).abs() < EPSILON);

        // For a uniform distribution on [2, 4]
        assert!((uniform_cdf(2.0, 2.0, 4.0).unwrap() - 0.0).abs() < EPSILON);
        assert!((uniform_cdf(3.0, 2.0, 4.0).unwrap() - 0.5).abs() < EPSILON);
        assert!((uniform_cdf(4.0, 2.0, 4.0).unwrap() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_cdf_outside_range() {
        // CDF should be 0 below the range
        assert!((uniform_cdf(-1.0, 0.0, 1.0).unwrap() - 0.0).abs() < EPSILON);

        // CDF should be 1 above the range
        assert!((uniform_cdf(2.0, 0.0, 1.0).unwrap() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_cdf_inverse_cdf_relationship() {
        // Test that CDF and inverse CDF are inverses of each other
        let a = 2.0;
        let b = 5.0;

        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let x = uniform_inverse_cdf(p, a, b).unwrap();
            let p_result = uniform_cdf(x, a, b).unwrap();
            assert!(
                (p - p_result).abs() < EPSILON,
                "CDF(inverse_CDF(p)) should equal p"
            );

            // Also verify that inverse_CDF(CDF(x)) ≈ x for points within the range
            if p > 0.0 && p < 1.0 {
                let x_within_range = a + p * (b - a);
                let p_cdf = uniform_cdf(x_within_range, a, b).unwrap();
                let x_result = uniform_inverse_cdf(p_cdf, a, b).unwrap();
                assert!(
                    (x_within_range - x_result).abs() < EPSILON,
                    "inverse_CDF(CDF(x)) should equal x"
                );
            }
        }
    }

    #[test]
    fn test_uniform_pdf_at_boundary_a() {
        // PDF at x == a should be 1/(b-a)
        let result = uniform_pdf(0.0, 0.0, 1.0).unwrap();
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_pdf_at_boundary_b() {
        // PDF at x == b should be 1/(b-a)
        let result = uniform_pdf(1.0, 0.0, 1.0).unwrap();
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_inverse_cdf_p_negative() {
        let result = uniform_inverse_cdf(-0.1, 0.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_uniform_inverse_cdf_p_greater_than_one() {
        let result = uniform_inverse_cdf(1.5, 0.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_uniform_inverse_cdf_p_zero() {
        let result = uniform_inverse_cdf(0.0, 2.0, 5.0).unwrap();
        // Should return a (lower bound)
        assert!((result - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_inverse_cdf_p_one() {
        let result = uniform_inverse_cdf(1.0, 2.0, 5.0).unwrap();
        // Should return b (upper bound)
        assert!((result - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_inverse_cdf_a_greater_than_b() {
        let result = uniform_inverse_cdf(0.5, 2.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_uniform_cdf_a_greater_than_b() {
        let result = uniform_cdf(0.5, 2.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_uniform_cdf_a_equal_b() {
        let result = uniform_cdf(0.5, 1.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_uniform_cdf_x_exactly_at_a() {
        // CDF at x == a should be 0
        let result = uniform_cdf(0.0, 0.0, 1.0).unwrap();
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_cdf_x_exactly_at_b() {
        // CDF at x == b should be 1
        let result = uniform_cdf(1.0, 0.0, 1.0).unwrap();
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_uniform_pdf_x_between_a_and_b() {
        // Test x strictly between a and b
        let result = uniform_pdf(0.5, 0.0, 1.0).unwrap();
        assert!((result - 1.0).abs() < EPSILON);
    }
}
