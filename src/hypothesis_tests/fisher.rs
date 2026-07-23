//! # Fisher's exact test (2×2 contingency tables)
//!
//! Exact test of independence for a 2×2 table — the tool of choice when
//! expected cell counts are too small for the chi-square approximation
//! (the usual rule of thumb: any expected count < 5). Matches
//! `scipy.stats.fisher_exact` conventions.

use crate::error::{StatsError, StatsResult};
use crate::hypothesis_tests::Alternative;
use crate::utils::special_functions::ln_gamma;

/// Result of Fisher's exact test.
#[derive(Debug, Clone, Copy)]
pub struct FisherExactResult {
    /// Sample odds ratio `(a·d)/(b·c)` (`inf` when `b·c = 0`).
    pub odds_ratio: f64,
    /// Exact p-value under the chosen [`Alternative`].
    pub p_value: f64,
}

/// ln C(n, k) for the hypergeometric weights.
fn ln_choose(n: u64, k: u64) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Fisher's exact test on the 2×2 table `[[a, b], [c, d]]`.
///
/// Conditions on the margins: under H₀ the count `a` follows a
/// hypergeometric distribution. `Alternative::TwoSided` sums the
/// probabilities of all tables at most as likely as the observed one
/// (scipy's definition); `Greater` tests for a *positive* association
/// (large `a`), `Less` for a negative one.
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::{Alternative, fisher_exact};
///
/// // Treatment: 8/10 responders — control: 1/6.
/// let res = fisher_exact(8, 2, 1, 5, Alternative::TwoSided).unwrap();
/// assert!(res.p_value < 0.05);
/// assert!((res.odds_ratio - 20.0).abs() < 1e-12);
/// ```
pub fn fisher_exact(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    alternative: Alternative,
) -> StatsResult<FisherExactResult> {
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let n = row1 + row2;
    if n == 0 {
        return Err(StatsError::invalid_input(
            "fisher_exact: the table must contain at least one observation",
        ));
    }

    let odds_ratio = if b * c == 0 {
        if a * d == 0 { f64::NAN } else { f64::INFINITY }
    } else {
        (a as f64 * d as f64) / (b as f64 * c as f64)
    };

    // Support of a: max(0, col1 − row2) ..= min(row1, col1).
    let k_min = col1.saturating_sub(row2);
    let k_max = row1.min(col1);

    // Hypergeometric log-PMF of each possible a, normalised by ln C(n, col1).
    let ln_norm = ln_choose(n, col1);
    let ln_pmf = |k: u64| ln_choose(row1, k) + ln_choose(row2, col1 - k) - ln_norm;

    let ln_p_obs = ln_pmf(a);
    let mut p_two = 0.0_f64;
    let mut p_less = 0.0_f64;
    let mut p_greater = 0.0_f64;
    // Relative slack for "at most as likely" comparisons (scipy uses the
    // same guard against floating-point noise on equal-probability tables).
    const REL_EPS: f64 = 1e-7;
    for k in k_min..=k_max {
        let lp = ln_pmf(k);
        let p = lp.exp();
        if lp <= ln_p_obs + REL_EPS {
            p_two += p;
        }
        if k <= a {
            p_less += p;
        }
        if k >= a {
            p_greater += p;
        }
    }

    let p_value = match alternative {
        Alternative::TwoSided => p_two,
        Alternative::Less => p_less,
        Alternative::Greater => p_greater,
    }
    .clamp(0.0, 1.0);

    Ok(FisherExactResult {
        odds_ratio,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_matches_scipy() {
        // Reference: scipy.stats.fisher_exact([[8, 2], [1, 5]]).
        let two = fisher_exact(8, 2, 1, 5, Alternative::TwoSided).unwrap();
        assert!((two.odds_ratio - 20.0).abs() < 1e-12);
        assert!((two.p_value - 0.034965034965034975).abs() < 1e-12);

        let less = fisher_exact(8, 2, 1, 5, Alternative::Less).unwrap();
        assert!((less.p_value - 0.9991258741258742).abs() < 1e-12);

        let greater = fisher_exact(8, 2, 1, 5, Alternative::Greater).unwrap();
        assert!((greater.p_value - 0.024475524475524483).abs() < 1e-12);
    }

    #[test]
    fn test_fisher_no_association() {
        // Perfectly balanced table → p = 1.
        let res = fisher_exact(5, 5, 5, 5, Alternative::TwoSided).unwrap();
        assert!((res.p_value - 1.0).abs() < 1e-10);
        assert!((res.odds_ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fisher_zero_cell() {
        let res = fisher_exact(10, 0, 0, 10, Alternative::TwoSided).unwrap();
        assert!(res.odds_ratio.is_infinite());
        assert!(res.p_value < 1e-4);
        assert!(fisher_exact(0, 0, 0, 0, Alternative::TwoSided).is_err());
    }
}
