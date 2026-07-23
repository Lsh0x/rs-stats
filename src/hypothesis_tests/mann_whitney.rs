//! # Mann-Whitney U test (Wilcoxon rank-sum)
//!
//! Non-parametric test of whether two independent samples come from the
//! same distribution. Uses the normal approximation with tie correction
//! and continuity correction — matching
//! `scipy.stats.mannwhitneyu(method="asymptotic")` (its default
//! `use_continuity=True`). Appropriate for `n ≳ 8` per group; below that,
//! prefer an exact-table lookup (not implemented here).

use crate::error::{StatsError, StatsResult};
use crate::hypothesis_tests::Alternative;
use crate::prob::erfc;
use crate::utils::numeric::average_ranks;
use num_traits::ToPrimitive;

/// Result of a Mann-Whitney U test.
#[derive(Debug, Clone, Copy)]
pub struct MannWhitneyResult {
    /// U statistic of the **first** sample (`U₁`), as in scipy.
    pub u_statistic: f64,
    /// Standard-normal z used for the p-value (continuity-corrected).
    pub z_score: f64,
    /// p-value under the chosen [`Alternative`].
    pub p_value: f64,
}

/// Standard normal survival function via erfc (exact in the tails).
fn norm_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2).unwrap_or(f64::NAN)
}

/// Mann-Whitney U test on two independent samples.
///
/// `Alternative::Less` means "`a` is stochastically smaller than `b`"
/// (scipy convention).
///
/// # Errors
/// Returns an error if either sample is empty or if all pooled values are
/// identical (the rank variance is then zero and the test is undefined).
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::{Alternative, mann_whitney_u};
///
/// let a = [1.5, 2.1, 3.3, 4.0];
/// let b = [3.9, 5.5, 6.6, 7.7];
/// let res = mann_whitney_u(&a, &b, Alternative::TwoSided).unwrap();
/// assert!(res.p_value < 0.15);
/// ```
pub fn mann_whitney_u<X, Y>(
    a: &[X],
    b: &[Y],
    alternative: Alternative,
) -> StatsResult<MannWhitneyResult>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::invalid_input(
            "mann_whitney_u: both samples must be non-empty",
        ));
    }
    let n1 = a.len();
    let n2 = b.len();
    let n = n1 + n2;

    let mut pooled: Vec<f64> = Vec::with_capacity(n);
    for v in a {
        pooled.push(v.to_f64().ok_or_else(|| {
            StatsError::conversion_error("mann_whitney_u: a not convertible to f64")
        })?);
    }
    for v in b {
        pooled.push(v.to_f64().ok_or_else(|| {
            StatsError::conversion_error("mann_whitney_u: b not convertible to f64")
        })?);
    }

    let ranks = average_ranks(&pooled);
    let r1: f64 = ranks[..n1].iter().sum();
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;

    // Tie-corrected variance of U:
    // σ² = n1·n2/12 · [(N+1) − Σ(t³−t)/(N(N−1))]
    let mut sorted = pooled;
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let mut tie_term = 0.0_f64;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        tie_term += t * t * t - t;
        i = j;
    }
    let nf = n as f64;
    let sigma_sq =
        (n1 * n2) as f64 / 12.0 * ((nf + 1.0) - tie_term / (nf * (nf - 1.0)));
    if sigma_sq <= 0.0 {
        return Err(StatsError::invalid_input(
            "mann_whitney_u: all pooled values are identical; the test is undefined",
        ));
    }
    let sigma = sigma_sq.sqrt();
    let mu = (n1 * n2) as f64 / 2.0;

    // Continuity correction of 0.5, applied toward the mean (scipy).
    let (z, p_value) = match alternative {
        Alternative::TwoSided => {
            let z = (u1.max(u2) - mu - 0.5) / sigma;
            (z, (2.0 * norm_sf(z)).min(1.0))
        }
        Alternative::Greater => {
            let z = (u1 - mu - 0.5) / sigma;
            (z, norm_sf(z))
        }
        Alternative::Less => {
            let z = (u2 - mu - 0.5) / sigma;
            (z, norm_sf(z))
        }
    };

    Ok(MannWhitneyResult {
        u_statistic: u1,
        z_score: z,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.mannwhitneyu(method='asymptotic').
    const A: [f64; 8] = [1.5, 2.1, 3.3, 4.0, 5.2, 6.1, 7.4, 8.0];
    const B: [f64; 9] = [2.0, 3.5, 4.1, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1];

    #[test]
    fn test_mann_whitney_two_sided() {
        let res = mann_whitney_u(&A, &B, Alternative::TwoSided).unwrap();
        assert!((res.u_statistic - 22.0).abs() < 1e-12);
        assert!((res.p_value - 0.19393085228241058).abs() < 1e-10);
    }

    #[test]
    fn test_mann_whitney_one_sided() {
        let less = mann_whitney_u(&A, &B, Alternative::Less).unwrap();
        assert!((less.p_value - 0.09696542614120529).abs() < 1e-10);
        let greater = mann_whitney_u(&A, &B, Alternative::Greater).unwrap();
        assert!((greater.p_value - 0.9185317500776713).abs() < 1e-10);
    }

    #[test]
    fn test_mann_whitney_ties() {
        let x = [1.0, 2.0, 2.0, 3.0, 4.0];
        let y = [2.0, 3.0, 3.0, 4.0, 5.0, 6.0];
        let res = mann_whitney_u(&x, &y, Alternative::TwoSided).unwrap();
        assert!((res.u_statistic - 6.5).abs() < 1e-12);
        assert!((res.p_value - 0.13585170221660398).abs() < 1e-10);
    }

    #[test]
    fn test_mann_whitney_degenerate() {
        assert!(mann_whitney_u::<f64, f64>(&[], &[1.0], Alternative::TwoSided).is_err());
        // All identical values → zero rank variance.
        let c = [3.0, 3.0, 3.0];
        assert!(mann_whitney_u(&c, &c, Alternative::TwoSided).is_err());
    }
}
