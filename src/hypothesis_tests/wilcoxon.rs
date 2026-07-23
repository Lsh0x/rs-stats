//! # Wilcoxon signed-rank test
//!
//! Non-parametric test for paired samples (or one sample against zero).
//! Uses the normal approximation with tie correction — matching
//! `scipy.stats.wilcoxon(zero_method="wilcox", correction=False,
//! method="approx")`. Appropriate for `n ≳ 10` non-zero pairs.

use crate::error::{StatsError, StatsResult};
use crate::hypothesis_tests::Alternative;
use crate::prob::erfc;
use crate::utils::numeric::average_ranks;
use num_traits::ToPrimitive;

/// Result of a Wilcoxon signed-rank test.
#[derive(Debug, Clone, Copy)]
pub struct WilcoxonResult {
    /// Sum of ranks of the positive differences (`W⁺`).
    pub w_plus: f64,
    /// Sum of ranks of the negative differences (`W⁻`).
    pub w_minus: f64,
    /// The test statistic as reported by scipy: `min(W⁺, W⁻)` for
    /// [`Alternative::TwoSided`], `W⁺` otherwise.
    pub statistic: f64,
    /// Standard-normal z used for the p-value.
    pub z_score: f64,
    /// p-value under the chosen [`Alternative`].
    pub p_value: f64,
    /// Number of non-zero pairs actually used.
    pub n_used: usize,
}

fn norm_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2).unwrap_or(f64::NAN)
}

/// Wilcoxon signed-rank test on paired samples.
///
/// Zero differences are dropped (`zero_method="wilcox"`), ties among the
/// absolute differences get average ranks with the usual variance
/// correction. `Alternative::Greater` means "`a` tends to be larger
/// than `b`".
///
/// # Errors
/// Returns an error on length mismatch, or when no non-zero difference
/// remains (all pairs equal).
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::{Alternative, wilcoxon_signed_rank};
///
/// let before = [125.0, 115.0, 130.0, 140.0, 140.0, 115.0, 140.0, 125.0, 140.0, 135.0];
/// let after  = [110.0, 122.0, 125.0, 120.0, 140.0, 124.0, 123.0, 137.0, 135.0, 145.0];
/// let res = wilcoxon_signed_rank(&before, &after, Alternative::TwoSided).unwrap();
/// assert!(res.p_value > 0.05);
/// ```
pub fn wilcoxon_signed_rank<X, Y>(
    a: &[X],
    b: &[Y],
    alternative: Alternative,
) -> StatsResult<WilcoxonResult>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    if a.len() != b.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "wilcoxon_signed_rank: samples must have the same length (got {} and {})",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(StatsError::invalid_input(
            "wilcoxon_signed_rank: samples must be non-empty",
        ));
    }

    // Differences, dropping zeros (zero_method = "wilcox").
    let mut diffs: Vec<f64> = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(b) {
        let xv = x.to_f64().ok_or_else(|| {
            StatsError::conversion_error("wilcoxon_signed_rank: a not convertible to f64")
        })?;
        let yv = y.to_f64().ok_or_else(|| {
            StatsError::conversion_error("wilcoxon_signed_rank: b not convertible to f64")
        })?;
        let d = xv - yv;
        if d != 0.0 {
            diffs.push(d);
        }
    }
    let n = diffs.len();
    if n == 0 {
        return Err(StatsError::invalid_input(
            "wilcoxon_signed_rank: all pairs are equal; the test is undefined",
        ));
    }

    let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = average_ranks(&abs_diffs);
    let w_plus: f64 = diffs
        .iter()
        .zip(&ranks)
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| r)
        .sum();
    let total = (n * (n + 1)) as f64 / 2.0;
    let w_minus = total - w_plus;

    // Tie-corrected variance: n(n+1)(2n+1)/24 − Σ(t³−t)/48.
    let mut sorted = abs_diffs;
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
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma_sq = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_term / 48.0;
    if sigma_sq <= 0.0 {
        return Err(StatsError::invalid_input(
            "wilcoxon_signed_rank: zero rank variance; the test is undefined",
        ));
    }
    let z = (w_plus - mu) / sigma_sq.sqrt();

    let (statistic, p_value) = match alternative {
        Alternative::TwoSided => (w_plus.min(w_minus), (2.0 * norm_sf(z.abs())).min(1.0)),
        Alternative::Greater => (w_plus, norm_sf(z)),
        Alternative::Less => (w_plus, 1.0 - norm_sf(z)),
    };

    Ok(WilcoxonResult {
        w_plus,
        w_minus,
        statistic,
        z_score: z,
        p_value,
        n_used: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.wilcoxon(method='approx').
    const A: [f64; 10] = [
        125.0, 115.0, 130.0, 140.0, 140.0, 115.0, 140.0, 125.0, 140.0, 135.0,
    ];
    const B: [f64; 10] = [
        110.0, 122.0, 125.0, 120.0, 140.0, 124.0, 123.0, 137.0, 135.0, 145.0,
    ];

    #[test]
    fn test_wilcoxon_two_sided() {
        let res = wilcoxon_signed_rank(&A, &B, Alternative::TwoSided).unwrap();
        // One zero pair is dropped → n = 9; statistic = min(W+, W−) = 18.
        assert_eq!(res.n_used, 9);
        assert!((res.statistic - 18.0).abs() < 1e-12);
        assert!((res.p_value - 0.5936305914425295).abs() < 1e-10);
    }

    #[test]
    fn test_wilcoxon_greater() {
        let res = wilcoxon_signed_rank(&A, &B, Alternative::Greater).unwrap();
        assert!((res.statistic - 27.0).abs() < 1e-12);
        assert!((res.p_value - 0.29681529572126475).abs() < 1e-10);
    }

    #[test]
    fn test_wilcoxon_degenerate() {
        let x = [1.0, 2.0, 3.0];
        assert!(wilcoxon_signed_rank(&x, &x, Alternative::TwoSided).is_err());
        assert!(wilcoxon_signed_rank(&x, &[1.0], Alternative::TwoSided).is_err());
    }
}
