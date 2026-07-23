//! # Two-sample Kolmogorov-Smirnov test
//!
//! Non-parametric test of whether two independent samples come from the
//! same continuous distribution, based on the maximum distance between
//! their empirical CDFs. Matches `scipy.stats.ks_2samp(method="asymp")`.
//!
//! For goodness-of-fit of one sample against a *known* distribution, use
//! [`crate::distributions::fitting::ks_test`].

use crate::distributions::fitting::{KsResult, kolmogorov_p};
use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;

/// Two-sample Kolmogorov-Smirnov test.
///
/// Returns the KS statistic `D = sup |F₁(x) − F₂(x)|` and the two-sided
/// p-value from the Kolmogorov distribution with the Numerical-Recipes
/// finite-sample correction (`(√nₑ + 0.12 + 0.11/√nₑ)·D`,
/// `nₑ = n₁n₂/(n₁+n₂)`) — the same convention as this crate's one-sample
/// [`ks_test`](crate::distributions::fitting::ks_test). The statistic
/// matches scipy exactly; the p-value can differ from
/// `scipy.stats.ks_2samp` by O(1/√n) (scipy applies its own finite-n
/// correction). Reasonable for `n₁, n₂ ≳ 10`.
///
/// # Errors
/// Returns an error if either sample is empty or contains values not
/// convertible to `f64`.
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::ks_2samp;
///
/// let a = [1.2, 2.4, 3.1, 4.8, 5.5, 6.1, 7.9, 8.2];
/// let b = [1.4, 2.1, 3.5, 4.2, 5.9, 6.6, 7.1, 8.8];
/// let res = ks_2samp(&a, &b).unwrap();
/// assert!(res.p_value > 0.5); // same underlying distribution
/// ```
pub fn ks_2samp<X, Y>(a: &[X], b: &[Y]) -> StatsResult<KsResult>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::invalid_input(
            "ks_2samp: both samples must be non-empty",
        ));
    }
    let mut xa: Vec<f64> = a
        .iter()
        .map(|v| v.to_f64())
        .collect::<Option<_>>()
        .ok_or_else(|| StatsError::conversion_error("ks_2samp: a not convertible to f64"))?;
    let mut xb: Vec<f64> = b
        .iter()
        .map(|v| v.to_f64())
        .collect::<Option<_>>()
        .ok_or_else(|| StatsError::conversion_error("ks_2samp: b not convertible to f64"))?;
    xa.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    xb.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let (n1, n2) = (xa.len(), xb.len());
    let (n1f, n2f) = (n1 as f64, n2 as f64);

    // Two-pointer sweep over the merged order; at each distinct value,
    // advance BOTH pointers past their ties before comparing the ECDFs
    // (a tie shared by the two samples is not a crossing point).
    let (mut i, mut j) = (0usize, 0usize);
    let mut d = 0.0_f64;
    while i < n1 && j < n2 {
        let x = xa[i].min(xb[j]);
        while i < n1 && xa[i] == x {
            i += 1;
        }
        while j < n2 && xb[j] == x {
            j += 1;
        }
        let diff = (i as f64 / n1f - j as f64 / n2f).abs();
        d = d.max(diff);
    }
    // Once one sample is exhausted the ECDF gap only shrinks back to 0 —
    // no further candidates.

    let sqrt_en = (n1f * n2f / (n1f + n2f)).sqrt();
    Ok(KsResult {
        statistic: d,
        p_value: kolmogorov_p((sqrt_en + 0.12 + 0.11 / sqrt_en) * d),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_2samp_matches_references() {
        let a = [
            1.5, 2.1, 3.3, 4.0, 5.2, 6.1, 7.4, 8.0, 9.3, 10.0, 11.2, 12.5,
        ];
        let b = [2.0, 3.5, 4.1, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 12.0, 13.5];
        let res = ks_2samp(&a, &b).unwrap();
        // Statistic matches scipy exactly.
        assert!((res.statistic - 0.1590909090909091).abs() < 1e-12);
        // p-value: NR-corrected Kolmogorov asymptotic
        // (kstwobign.sf((√nₑ+0.12+0.11/√nₑ)·D) = 0.99635…); scipy's own
        // finite-n correction gives 0.99129 — same conclusion either way.
        assert!((res.p_value - 0.9963463314278793).abs() < 1e-9);
        assert!((res.p_value - 0.9912889825608985).abs() < 0.02);
    }

    #[test]
    fn test_ks_2samp_shifted_distributions() {
        let a: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..50).map(|i| i as f64 * 0.1 + 3.0).collect();
        let res = ks_2samp(&a, &b).unwrap();
        assert!(res.statistic > 0.5);
        assert!(res.p_value < 1e-6);
    }

    #[test]
    fn test_ks_2samp_empty() {
        assert!(ks_2samp::<f64, f64>(&[], &[1.0]).is_err());
    }
}
