//! # Binomial Distribution
//!
//! This module implements the Binomial distribution, a discrete probability distribution
//! that models the number of successes in a sequence of independent experiments.
//!
//! ## Key Characteristics
//! - Models the number of successes in `n` independent trials
//! - Each trial has success probability `p`
//! - Discrete probability distribution
//!
//! ## Common Applications
//! - Quality control testing
//! - A/B testing
//! - Risk analysis
//! - Genetics (Mendelian inheritance)
//!
//! ## Mathematical Formulation
//! The probability mass function (PMF) is given by:
//!
//! P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
//!
//! where:
//! - n is the number of trials
//! - k is the number of successes
//! - p is the probability of success
//! - C(n,k) is the binomial coefficient (n choose k)

use crate::error::{StatsError, StatsResult};
use crate::utils::special_functions::{ln_gamma, regularized_incomplete_beta};
// Private math helpers; the public API is the [`Binomial`] struct's
// [`DiscreteDistribution`] impl below.

/// Probability mass function (PMF) for the Binomial distribution.
///
/// Calculates the probability of observing exactly `k` successes in `n` trials
/// with success probability `p`.
///
/// # Arguments
/// * `k` - The number of successes (must be ≤ n)
/// * `n` - The total number of trials (must be positive)
/// * `p` - The probability of success in a single trial (must be between 0 and 1)
///
/// # Returns
/// The probability of exactly `k` successes occurring.
///
/// # Errors
/// Returns an error if:
/// - n is zero
/// - p is not between 0 and 1
/// - k > n
/// - Type conversion to f64 fails
///
#[inline]
fn pmf(k: u64, n: u64, p: f64) -> StatsResult<f64> {
    if n == 0 {
        return Err(StatsError::InvalidInput {
            message: "binomial::pmf: n must be positive".to_string(),
        });
    }
    if !((0.0..=1.0).contains(&p)) {
        return Err(StatsError::InvalidInput {
            message: "binomial::pmf: p must be between 0 and 1".to_string(),
        });
    }
    if k > n {
        return Err(StatsError::InvalidInput {
            message: "binomial::pmf: k must be less than or equal to n".to_string(),
        });
    }
    if p == 0.0 {
        return Ok(if k == 0 { 1.0 } else { 0.0 });
    }
    if p == 1.0 {
        return Ok(if k == n { 1.0 } else { 0.0 });
    }
    // Entirely in log-space: a linear-space C(n, k) overflows f64 (→ inf,
    // then inf·exp(…) = NaN) as soon as n ≈ 1030.
    let log_binom =
        ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64);
    let log_prob = k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln();
    Ok((log_binom + log_prob).exp())
}

/// Cumulative distribution function (CDF) for the Binomial distribution.
///
/// Calculates the probability of observing `k` or fewer successes in `n` trials
/// with success probability `p`.
///
/// # Arguments
/// * `k` - The maximum number of successes (must be ≤ n)
/// * `n` - The total number of trials (must be positive)
/// * `p` - The probability of success in a single trial (must be between 0 and 1)
///
/// # Returns
/// The cumulative probability of `k` or fewer successes occurring.
///
/// # Errors
/// Returns an error if:
/// - n is zero
/// - p is not between 0 and 1
/// - k > n
/// - Type conversion to f64 fails
///
#[inline]
fn cdf(k: u64, n: u64, p: f64) -> StatsResult<f64> {
    if n == 0 {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::cdf: n must be positive".to_string(),
        });
    }
    if !((0.0..=1.0).contains(&p)) {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::cdf: p must be between 0 and 1".to_string(),
        });
    }
    if k > n {
        return Err(StatsError::InvalidInput {
            message: "binomial_distribution::cdf: k must be less than or equal to n".to_string(),
        });
    }
    if p == 0.0 {
        return Ok(1.0); // P(X <= k) = 1 when p = 0 (all mass at k=0)
    }
    if p == 1.0 {
        return Ok(if k >= n { 1.0 } else { 0.0 });
    }
    if k == n {
        return Ok(1.0);
    }
    // Closed form via the regularized incomplete beta:
    //   P(X ≤ k) = I_{1−p}(n−k, k+1)
    // O(1) and stable for any n. The previous O(k) PMF recurrence started at
    // (1−p)^n, which underflows to 0 for n ≳ 1000 — the whole sum then
    // silently collapsed to 0.
    Ok(regularized_incomplete_beta((n - k) as f64, (k + 1) as f64, 1.0 - p).clamp(0.0, 1.0))
}

// ── Typed struct + DiscreteDistribution impl ───────────────────────────────────

/// Binomial distribution Binomial(n, p) as a typed struct.
///
/// # Examples
/// ```
/// use rs_stats::distributions::binomial_distribution::Binomial;
/// use rs_stats::distributions::traits::Distribution;
///
/// let b = Binomial::new(10, 0.5).unwrap();
/// assert!((b.mean() - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Binomial {
    /// Number of trials n (must be ≥ 1)
    pub n: u64,
    /// Success probability p ∈ [0, 1]
    pub p: f64,
}

impl Binomial {
    /// Creates a `Binomial` distribution with validation.
    pub fn new(n: u64, p: f64) -> StatsResult<Self> {
        if n == 0 {
            return Err(StatsError::InvalidInput {
                message: "Binomial::new: n must be at least 1".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidInput {
                message: "Binomial::new: p must be in [0, 1]".to_string(),
            });
        }
        Ok(Self { n, p })
    }

    /// MLE: assume n = max(data), p = mean(data) / n.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Binomial::fit: data must not be empty".to_string(),
            });
        }
        let n = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .round() as u64;
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let p = if n == 0 { 0.5 } else { mean / n as f64 };
        Self::new(n.max(1), p.clamp(0.0, 1.0))
    }
}

impl crate::distributions::traits::Distribution for Binomial {
    type X = u64;
    fn name(&self) -> &str {
        "Binomial"
    }
    fn num_params(&self) -> usize {
        2
    }
    fn pdf(&self, k: u64) -> StatsResult<f64> {
        pmf(k, self.n, self.p)
    }
    /// Log-space PMF for numerical stability with large n or k.
    ///
    /// ln P(X=k) = ln Γ(n+1) − ln Γ(k+1) − ln Γ(n−k+1) + k·ln(p) + (n−k)·ln(1−p)
    fn logpdf(&self, k: u64) -> StatsResult<f64> {
        let n = self.n;
        if k > n {
            return Ok(f64::NEG_INFINITY);
        }
        // ln C(n,k) via ln_gamma — exact and stable for any n, k.
        let log_binom =
            ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64);
        let log_p = match (self.p, k) {
            (0.0, 0) => 0.0,
            (0.0, _) => return Ok(f64::NEG_INFINITY),
            (_, _) => k as f64 * self.p.ln(),
        };
        let log_q = match (self.p, n - k) {
            (1.0, 0) => 0.0,
            (1.0, _) => return Ok(f64::NEG_INFINITY),
            (_, nk) => nk as f64 * (1.0 - self.p).ln(),
        };
        Ok(log_binom + log_p + log_q)
    }
    fn cdf(&self, k: u64) -> StatsResult<f64> {
        cdf(k, self.n, self.p)
    }
    /// Exact tail: `S(k) = P(X ≥ k+1) = I_p(k+1, n−k)`.
    fn sf(&self, k: u64) -> StatsResult<f64> {
        if k >= self.n {
            return Ok(0.0);
        }
        if self.p == 0.0 {
            return Ok(0.0);
        }
        if self.p == 1.0 {
            return Ok(1.0); // k < n here
        }
        Ok(
            regularized_incomplete_beta((k + 1) as f64, (self.n - k) as f64, self.p)
                .clamp(0.0, 1.0),
        )
    }
    /// Direct sampling: BINV sequential inversion (mean `n·min(p,1−p)`
    /// steps of a multiplicative recurrence) when that mean is ≤ 30,
    /// otherwise the generic quantile search. Symmetry `k ↔ n−k` keeps
    /// the walk on the cheap side of p.
    fn sample(&self, rng: &mut dyn rand::RngCore) -> StatsResult<u64> {
        use crate::distributions::normal_distribution::uniform01;
        let (pp, flipped) = if self.p <= 0.5 {
            (self.p, false)
        } else {
            (1.0 - self.p, true)
        };
        let n = self.n;
        let k = if pp == 0.0 {
            0
        } else if (n as f64) * pp <= 30.0 {
            // BINV: walk the PMF from k = 0 with the standard recurrence.
            let q = 1.0 - pp;
            let s = pp / q;
            let mut f = q.powf(n as f64); // pmf(0)
            let mut u = uniform01(rng);
            let mut k = 0u64;
            while u > f && k < n {
                u -= f;
                k += 1;
                f *= s * ((n - k + 1) as f64) / (k as f64);
            }
            k
        } else {
            // Large n·p: exact quantile search on the flipped parameter.
            let d = Binomial { n, p: pp };
            crate::distributions::traits::discrete_inverse_cdf_search(uniform01(rng), |k| {
                if k >= n { Ok(1.0) } else { d.cdf(k) }
            })?
        };
        Ok(if flipped { n - k } else { k })
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<u64> {
        // Bounded support: p = 1 has the finite quantile n, and the search's
        // doubling phase may probe k > n, where `cdf` errors — clamp to 1.
        if p == 1.0 {
            return Ok(self.n);
        }
        crate::distributions::traits::discrete_inverse_cdf_search(p, |k| {
            if k >= self.n { Ok(1.0) } else { self.cdf(k) }
        })
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_pmf() {
        let n = 10;
        let p = 0.5;
        let k = 5;
        let result = pmf(k, n, p).unwrap();
        assert!(
            !result.is_nan(),
            "PMF returned NaN for k={}, n={}, p={}",
            k,
            n,
            p
        );
    }

    #[test]
    fn test_binomial_cdf() {
        let n = 10;
        let p = 0.5;
        let k = 5;
        let result = cdf(k, n, p).unwrap();
        assert!(
            !result.is_nan(),
            "CDF returned NaN for k={}, n={}, p={}",
            k,
            n,
            p
        );
    }

    #[test]
    fn test_binomial_pmf_large_values_n() {
        // Test with values that exceed i32::MAX to verify overflow protection
        // Using values just above i32::MAX (2,147,483,647)
        let n = 2_200_000_000u64;
        let k = 5u64;
        let p = 0.5;

        // This should not panic or truncate - should use powf() path
        let result = pmf(k, n, p);

        // Result might be very small or NaN due to numerical precision, but shouldn't panic
        match result {
            Ok(val) => {
                // Value should be valid (might be very small due to large n)
                assert!(
                    !val.is_infinite(),
                    "PMF should not be infinite for large values"
                );
            }
            Err(_) => {
                // Error is acceptable for very large values (numerical precision limits)
            }
        }
    }

    #[test]
    fn test_binomial_pmf_large_values_k() {
        // Test with values that exceed i32::MAX to verify overflow protection
        // Using values just above i32::MAX (2,147,483,647)
        let n = 2u64;
        let k = 2_200_000_000_000u64;
        let p = 0.5;

        // This should not panic or truncate - should use powf() path
        let result = pmf(k, n, p);

        // Result might be very small or NaN due to numerical precision, but shouldn't panic
        match result {
            Ok(val) => {
                // Value should be valid (might be very small due to large n)
                assert!(
                    !val.is_infinite(),
                    "PMF should not be infinite for large values"
                );
            }
            Err(_) => {
                // Error is acceptable for very large values (numerical precision limits)
            }
        }
    }

    #[test]
    fn test_binomial_pmf_p_zero_k_zero() {
        // When p=0.0 and k=0, PMF should return combinations (which is 1 for k=0)
        let result = pmf(0, 10, 0.0).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_binomial_pmf_p_zero_k_greater_than_zero() {
        // When p=0.0 and k>0, PMF should return 0.0
        let result = pmf(5, 10, 0.0).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_binomial_pmf_p_one_k_equals_n() {
        // When p=1.0 and k=n, PMF should return combinations (which is 1 for k=n)
        let result = pmf(10, 10, 1.0).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_binomial_pmf_p_one_k_less_than_n() {
        // When p=1.0 and k<n, PMF should return 0.0
        let result = pmf(5, 10, 1.0).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_binomial_pmf_n_zero() {
        let result = pmf(0, 0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_pmf_p_out_of_range() {
        let result = pmf(5, 10, 1.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_cdf_k_greater_than_n() {
        let result = cdf(15, 10, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_pmf_large_n_no_nan() {
        // n ≈ 1030 used to overflow C(n, k) to inf in linear space,
        // producing pmf = inf·exp(−large) = NaN. Log-space keeps it finite.
        for (n, p, k) in [
            (1030_u64, 0.5_f64, 515_u64),
            (1500, 0.3, 450),
            (5000, 0.5, 2500),
        ] {
            let v = pmf(k, n, p).unwrap();
            assert!(v.is_finite() && v > 0.0, "pmf({k}, {n}, {p}) = {v}");
            // Consistent with the log-space trait path.
            use crate::distributions::traits::Distribution;
            let d = Binomial::new(n, p).unwrap();
            let via_log = d.logpdf(k).unwrap().exp();
            assert!((v - via_log).abs() <= 1e-12 * via_log);
        }
    }

    #[test]
    fn test_binomial_pmf_k_greater_than_n() {
        // When k > n, combination() should return an error
        let result = pmf(15, 10, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_cdf_n_zero() {
        let result = cdf(5, 0, 0.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_binomial_cdf_p_out_of_range() {
        let result = cdf(5, 10, 1.5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }
}
