//! # Statistical power & sample-size calculations for t-tests
//!
//! "How many observations do I need to detect an effect of size d?" —
//! answered **exactly** through the noncentral t distribution (like
//! G*Power and `statsmodels.stats.power`), not a normal approximation.
//!
//! Effect size is Cohen's d (0.2 small, 0.5 medium, 0.8 large); see
//! [`cohens_d`](crate::hypothesis_tests::cohens_d) to estimate it from
//! pilot data.

use crate::distributions::student_t::StudentT;
use crate::distributions::traits::Distribution as _;
use crate::error::{StatsError, StatsResult};
use crate::hypothesis_tests::Alternative;
use crate::utils::special_functions::{noncentral_t_cdf, noncentral_t_sf};

/// Which t-test the power calculation refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TTestKind {
    /// One sample against a fixed mean; `n` = number of observations.
    OneSample,
    /// Two independent samples (equal sizes); `n` = observations **per group**.
    TwoSample,
    /// Paired samples; `n` = number of pairs, `effect_size` = d of the differences.
    Paired,
}

fn df_and_ncp(kind: TTestKind, n: u64, d: f64) -> (f64, f64) {
    let nf = n as f64;
    match kind {
        TTestKind::TwoSample => (2.0 * nf - 2.0, d * (nf / 2.0).sqrt()),
        TTestKind::OneSample | TTestKind::Paired => (nf - 1.0, d * nf.sqrt()),
    }
}

/// Every design needs at least 2 observations (df ≥ 1).
const MIN_N: u64 = 2;

/// Exact power of a t-test: the probability of rejecting H₀ at level
/// `alpha` when the true standardized effect is `effect_size`.
///
/// # Errors
/// Returns an error for `n` too small (df ≤ 0), `alpha` outside (0, 1),
/// or a non-finite effect size.
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::{power_t_test, Alternative, TTestKind};
///
/// // 30 subjects per arm, medium effect: badly underpowered.
/// let p = power_t_test(TTestKind::TwoSample, 30, 0.5, 0.05, Alternative::TwoSided).unwrap();
/// assert!((p - 0.478).abs() < 0.001);
/// ```
pub fn power_t_test(
    kind: TTestKind,
    n: u64,
    effect_size: f64,
    alpha: f64,
    alternative: Alternative,
) -> StatsResult<f64> {
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatsError::invalid_parameter(format!(
            "power_t_test: alpha must be in (0, 1), got {alpha}"
        )));
    }
    if !effect_size.is_finite() {
        return Err(StatsError::invalid_input(
            "power_t_test: effect_size must be finite",
        ));
    }
    if n < MIN_N {
        return Err(StatsError::invalid_input(format!(
            "power_t_test: n = {n} is too small for this design"
        )));
    }

    let (df, ncp) = df_and_ncp(kind, n, effect_size);
    let t = StudentT::new(0.0, 1.0, df)?;
    let power = match alternative {
        Alternative::TwoSided => {
            let t_crit = t.inverse_cdf(1.0 - alpha / 2.0)?;
            noncentral_t_sf(t_crit, df, ncp) + noncentral_t_cdf(-t_crit, df, ncp)
        }
        Alternative::Greater => {
            let t_crit = t.inverse_cdf(1.0 - alpha)?;
            noncentral_t_sf(t_crit, df, ncp)
        }
        Alternative::Less => {
            let t_crit = t.inverse_cdf(alpha)?;
            noncentral_t_cdf(t_crit, df, ncp)
        }
    };
    Ok(power.clamp(0.0, 1.0))
}

/// Smallest `n` (per group for [`TTestKind::TwoSample`]) reaching the
/// target `power` for the given effect size and level.
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::{sample_size_t_test, Alternative, TTestKind};
///
/// // The classic: d = 0.5, 80% power, two-sided α = 0.05 → 64 per group.
/// let n = sample_size_t_test(TTestKind::TwoSample, 0.5, 0.05, 0.8, Alternative::TwoSided).unwrap();
/// assert_eq!(n, 64);
/// ```
pub fn sample_size_t_test(
    kind: TTestKind,
    effect_size: f64,
    alpha: f64,
    power: f64,
    alternative: Alternative,
) -> StatsResult<u64> {
    if !(0.0 < power && power < 1.0) {
        return Err(StatsError::invalid_parameter(format!(
            "sample_size_t_test: power must be in (0, 1), got {power}"
        )));
    }
    if effect_size == 0.0 || !effect_size.is_finite() {
        return Err(StatsError::invalid_input(
            "sample_size_t_test: effect_size must be non-zero and finite",
        ));
    }

    // Exponential search for an upper bound, then binary search for the
    // minimal n. Power is monotone in n for a fixed effect.
    let mut lo = MIN_N;
    let mut hi = lo;
    while power_t_test(kind, hi, effect_size, alpha, alternative)? < power {
        hi = hi.saturating_mul(2);
        if hi > 100_000_000 {
            return Err(StatsError::invalid_input(
                "sample_size_t_test: required n exceeds 1e8 — effect size too small",
            ));
        }
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if power_t_test(kind, mid, effect_size, alpha, alternative)? < power {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::special_functions::noncentral_t_cdf as nct_cdf;

    #[test]
    fn test_noncentral_t_matches_scipy() {
        // Reference: scipy.stats.nct.cdf.
        for (t, df, nc, expected) in [
            (1.5, 10.0, 2.0, 0.3047854473760423),
            (-0.5, 5.0, 1.0, 0.07230224347690409),
            (3.0, 20.0, 2.5, 0.6616028734935053),
            (0.0, 8.0, 0.5, 0.3085375387259869),
            (2.0, 3.0, -1.5, 0.9971191699279385),
        ] {
            let got = nct_cdf(t, df, nc);
            assert!(
                (got - expected).abs() < 1e-10,
                "nct_cdf({t}, {df}, {nc}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_power_matches_scipy() {
        // References computed with scipy.stats.nct (same as statsmodels).
        let p = power_t_test(TTestKind::TwoSample, 30, 0.5, 0.05, Alternative::TwoSided).unwrap();
        assert!((p - 0.4778965207601648).abs() < 1e-9, "{p}");
        let p = power_t_test(TTestKind::TwoSample, 64, 0.5, 0.05, Alternative::TwoSided).unwrap();
        assert!((p - 0.8014595579222545).abs() < 1e-9, "{p}");
        let p = power_t_test(TTestKind::OneSample, 20, 0.6, 0.05, Alternative::TwoSided).unwrap();
        assert!((p - 0.7210050995594134).abs() < 1e-9, "{p}");
        let p = power_t_test(TTestKind::TwoSample, 20, 0.5, 0.05, Alternative::Greater).unwrap();
        assert!((p - 0.4633743492964092).abs() < 1e-9, "{p}");
    }

    #[test]
    fn test_sample_size_classic() {
        // d = 0.5, 80% power, α = 0.05 two-sided → 64/group (statsmodels: 63.77).
        let n = sample_size_t_test(TTestKind::TwoSample, 0.5, 0.05, 0.8, Alternative::TwoSided)
            .unwrap();
        assert_eq!(n, 64);
        // Boundary sanity: n−1 must be underpowered.
        let p63 = power_t_test(TTestKind::TwoSample, 63, 0.5, 0.05, Alternative::TwoSided).unwrap();
        assert!(p63 < 0.8);
    }

    #[test]
    fn test_errors() {
        assert!(power_t_test(TTestKind::TwoSample, 1, 0.5, 0.05, Alternative::TwoSided).is_err());
        assert!(power_t_test(TTestKind::TwoSample, 30, 0.5, 1.5, Alternative::TwoSided).is_err());
        assert!(
            sample_size_t_test(TTestKind::TwoSample, 0.0, 0.05, 0.8, Alternative::TwoSided)
                .is_err()
        );
    }
}
