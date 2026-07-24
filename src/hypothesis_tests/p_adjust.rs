//! # Multiple-comparison corrections
//!
//! When you run *m* tests, the chance of at least one false positive at
//! α = 0.05 is `1 − 0.95^m` — 40% for just 10 tests. [`p_adjust`] rescales
//! p-values so the usual `p < α` decision rule stays valid across the
//! whole family of tests. Same conventions as R's `p.adjust` /
//! `scipy.stats.false_discovery_control`.

use crate::error::{StatsError, StatsResult};

/// Correction method for [`p_adjust`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PAdjustMethod {
    /// Bonferroni: `p·m`, clipped at 1. Controls the family-wise error
    /// rate (FWER); simplest and most conservative.
    Bonferroni,
    /// Holm step-down: uniformly more powerful than Bonferroni while
    /// controlling the same FWER. A sensible default.
    #[default]
    Holm,
    /// Benjamini-Hochberg step-up: controls the false-discovery rate
    /// (FDR) instead of the FWER — more discoveries when many tests are
    /// expected to be non-null (screening, omics, A/B batteries).
    BenjaminiHochberg,
}

/// Adjust a family of p-values for multiple comparisons.
///
/// Returns adjusted p-values in the **same order** as the input; compare
/// them to your α directly.
///
/// # Errors
/// Returns an error if `p_values` is empty or contains values outside
/// `[0, 1]` (or NaN).
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::{p_adjust, PAdjustMethod};
///
/// let p = [0.01, 0.04, 0.03, 0.005, 0.2, 0.5];
/// let adj = p_adjust(&p, PAdjustMethod::Holm).unwrap();
/// // Only the smallest raw p survives at α = 0.05 (vs four raw ones).
/// assert_eq!(adj.iter().filter(|&&q| q < 0.05).count(), 1);
/// assert!((adj[3] - 0.03).abs() < 1e-12);
/// ```
pub fn p_adjust(p_values: &[f64], method: PAdjustMethod) -> StatsResult<Vec<f64>> {
    if p_values.is_empty() {
        return Err(StatsError::empty_data(
            "p_adjust: p_values must not be empty",
        ));
    }
    if p_values.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err(StatsError::invalid_input(
            "p_adjust: all p-values must be in [0, 1]",
        ));
    }

    let m = p_values.len();
    let mf = m as f64;
    let mut order: Vec<usize> = (0..m).collect();
    let mut adjusted = vec![0.0_f64; m];

    match method {
        PAdjustMethod::Bonferroni => {
            for (a, &p) in adjusted.iter_mut().zip(p_values) {
                *a = (p * mf).min(1.0);
            }
        }
        PAdjustMethod::Holm => {
            // Ascending; running max of (m − rank)·p keeps monotonicity.
            order.sort_by(|&a, &b| {
                p_values[a]
                    .partial_cmp(&p_values[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut running_max = 0.0_f64;
            for (rank, &i) in order.iter().enumerate() {
                running_max = running_max.max((mf - rank as f64) * p_values[i]);
                adjusted[i] = running_max.min(1.0);
            }
        }
        PAdjustMethod::BenjaminiHochberg => {
            // Descending; running min of p·m/rank keeps monotonicity.
            order.sort_by(|&a, &b| {
                p_values[b]
                    .partial_cmp(&p_values[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut running_min = 1.0_f64;
            for (idx, &i) in order.iter().enumerate() {
                let rank = (m - idx) as f64; // m, m−1, …, 1
                running_min = running_min.min(p_values[i] * mf / rank);
                adjusted[i] = running_min;
            }
        }
    }

    Ok(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: [f64; 6] = [0.01, 0.04, 0.03, 0.005, 0.2, 0.5];

    fn close(a: &[f64], b: &[f64]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn test_bonferroni() {
        let adj = p_adjust(&P, PAdjustMethod::Bonferroni).unwrap();
        close(&adj, &[0.06, 0.24, 0.18, 0.03, 1.0, 1.0]);
    }

    #[test]
    fn test_holm_matches_r() {
        // Reference: R p.adjust(method = "holm").
        let adj = p_adjust(&P, PAdjustMethod::Holm).unwrap();
        close(&adj, &[0.05, 0.12, 0.12, 0.03, 0.4, 0.5]);
    }

    #[test]
    fn test_bh_matches_scipy() {
        // Reference: scipy.stats.false_discovery_control(method='bh').
        let adj = p_adjust(&P, PAdjustMethod::BenjaminiHochberg).unwrap();
        close(&adj, &[0.03, 0.06, 0.06, 0.03, 0.24, 0.5]);
    }

    #[test]
    fn test_errors_and_edge() {
        assert!(p_adjust(&[], PAdjustMethod::Holm).is_err());
        assert!(p_adjust(&[0.5, 1.2], PAdjustMethod::Holm).is_err());
        assert!(p_adjust(&[0.5, f64::NAN], PAdjustMethod::Holm).is_err());
        // Single test: no correction.
        let adj = p_adjust(&[0.04], PAdjustMethod::Holm).unwrap();
        assert!((adj[0] - 0.04).abs() < 1e-15);
    }
}
