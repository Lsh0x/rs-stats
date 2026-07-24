//! # Resampling: bootstrap confidence intervals & permutation tests
//!
//! Distribution-free inference for **any** statistic — no normality
//! assumption, no closed-form standard error needed. Pass a closure, get
//! an interval or a p-value.
//!
//! ```
//! use rs_stats::resampling::bootstrap_ci;
//! use rs_stats::prob::quantile;
//! use rand::SeedableRng;
//!
//! let data = [12.1, 14.8, 15.2, 17.9, 19.3, 22.4, 25.1, 28.7, 33.0, 41.2,
//!             48.9, 55.3, 71.8, 13.4, 16.0, 18.2, 12.9, 14.1, 96.5, 24.6];
//! let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
//! // 95% CI for the MEDIAN — no formula exists; the bootstrap doesn't care.
//! let ci = bootstrap_ci(&data, |s| quantile(s, 0.5).unwrap(), 2000, 0.95, &mut rng).unwrap();
//! assert!(ci.lower < ci.estimate && ci.estimate < ci.upper);
//! ```

use crate::error::{StatsError, StatsResult};
use crate::hypothesis_tests::Alternative;
use rand::RngCore;

/// Result of [`bootstrap_ci`].
#[derive(Debug, Clone, Copy)]
pub struct BootstrapCi {
    /// The statistic evaluated on the original data.
    pub estimate: f64,
    /// Lower bound of the percentile confidence interval.
    pub lower: f64,
    /// Upper bound of the percentile confidence interval.
    pub upper: f64,
    /// Bootstrap standard error (sample std-dev of the replicates).
    pub std_error: f64,
    /// Number of resamples actually used.
    pub n_resamples: usize,
}

/// Draw an index in `[0, n)` from the top 53 bits of a `u64`.
/// The modulo bias is ~n/2⁵³ — irrelevant for any realistic n.
#[inline]
fn index(rng: &mut dyn RngCore, n: usize) -> usize {
    ((rng.next_u64() >> 11) as usize) % n
}

/// Percentile-bootstrap confidence interval for an arbitrary statistic.
///
/// Resamples `data` with replacement `n_resamples` times, evaluates
/// `statistic` on each replicate, and returns the empirical
/// `(1−level)/2` and `(1+level)/2` quantiles of the replicates (the
/// *percentile* method — same default flavour as `scipy.stats.bootstrap`
/// offers). Prefer ≥ 2000 resamples for a 95% interval.
///
/// # Errors
/// Returns an error on empty data, `n_resamples < 100`, or `level`
/// outside (0, 1).
pub fn bootstrap_ci(
    data: &[f64],
    statistic: impl Fn(&[f64]) -> f64,
    n_resamples: usize,
    level: f64,
    rng: &mut dyn RngCore,
) -> StatsResult<BootstrapCi> {
    if data.is_empty() {
        return Err(StatsError::empty_data(
            "bootstrap_ci: data must not be empty",
        ));
    }
    if n_resamples < 100 {
        return Err(StatsError::invalid_parameter(format!(
            "bootstrap_ci: n_resamples must be ≥ 100, got {n_resamples}"
        )));
    }
    if !(0.0 < level && level < 1.0) {
        return Err(StatsError::invalid_parameter(format!(
            "bootstrap_ci: level must be in (0, 1), got {level}"
        )));
    }

    let n = data.len();
    let estimate = statistic(data);

    let mut scratch = vec![0.0_f64; n];
    let mut replicates = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        for slot in scratch.iter_mut() {
            *slot = data[index(rng, n)];
        }
        replicates.push(statistic(&scratch));
    }

    let mean = replicates.iter().sum::<f64>() / n_resamples as f64;
    let var = replicates
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f64>()
        / (n_resamples - 1) as f64;

    replicates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| {
        let h = (n_resamples - 1) as f64 * p;
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        replicates[lo] + (h - lo as f64) * (replicates[hi] - replicates[lo])
    };
    let alpha = (1.0 - level) / 2.0;

    Ok(BootstrapCi {
        estimate,
        lower: q(alpha),
        upper: q(1.0 - alpha),
        std_error: var.sqrt(),
        n_resamples,
    })
}

/// Result of [`permutation_test`].
#[derive(Debug, Clone, Copy)]
pub struct PermutationTest {
    /// The statistic on the original (unpermuted) samples.
    pub statistic: f64,
    /// Monte-Carlo p-value, `(1 + #extreme) / (n_permutations + 1)`.
    pub p_value: f64,
    /// Number of random permutations drawn.
    pub n_permutations: usize,
}

/// Two-sample permutation test for an arbitrary statistic.
///
/// Under H₀ ("the two samples come from the same distribution"), group
/// labels are exchangeable: the pooled values are reshuffled
/// `n_permutations` times and `statistic(a', b')` recomputed on each
/// split. The p-value uses the add-one estimator
/// `(1 + #extreme)/(B + 1)`, which is never exactly 0.
///
/// [`Alternative::TwoSided`] compares `|statistic|`, so it assumes the
/// statistic is centred at 0 under H₀ (e.g. a difference of means or
/// medians). `Greater` / `Less` compare the signed value.
///
/// # Examples
/// ```
/// use rs_stats::resampling::permutation_test;
/// use rs_stats::hypothesis_tests::Alternative;
/// use rand::SeedableRng;
///
/// let a = [1.2, 1.5, 1.1, 1.8, 1.3, 1.6, 1.4, 1.7];
/// let b = [2.4, 2.1, 2.8, 2.3, 2.6, 2.2, 2.7, 2.5];
/// let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
/// let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
/// let r = permutation_test(&a, &b, |x, y| mean(x) - mean(y), 9999,
///                          Alternative::TwoSided, &mut rng).unwrap();
/// assert!(r.p_value < 0.001);
/// ```
pub fn permutation_test(
    a: &[f64],
    b: &[f64],
    statistic: impl Fn(&[f64], &[f64]) -> f64,
    n_permutations: usize,
    alternative: Alternative,
    rng: &mut dyn RngCore,
) -> StatsResult<PermutationTest> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::invalid_input(
            "permutation_test: both samples must be non-empty",
        ));
    }
    if n_permutations < 100 {
        return Err(StatsError::invalid_parameter(format!(
            "permutation_test: n_permutations must be ≥ 100, got {n_permutations}"
        )));
    }

    let observed = statistic(a, b);
    let n_a = a.len();
    let mut pool: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let n = pool.len();

    // Relative slack against ties lost to floating-point noise (as scipy).
    let gamma = 1e-14 * observed.abs().max(1.0);
    let mut extreme = 0usize;
    for _ in 0..n_permutations {
        // Partial Fisher-Yates: after n_a swaps, pool[..n_a] is a uniform
        // random subset in random order — exactly a label permutation.
        for i in 0..n_a {
            let j = i + index(rng, n - i);
            pool.swap(i, j);
        }
        let t = statistic(&pool[..n_a], &pool[n_a..]);
        let hit = match alternative {
            Alternative::TwoSided => t.abs() >= observed.abs() - gamma,
            Alternative::Greater => t >= observed - gamma,
            Alternative::Less => t <= observed + gamma,
        };
        if hit {
            extreme += 1;
        }
    }

    Ok(PermutationTest {
        statistic: observed,
        p_value: (1 + extreme) as f64 / (n_permutations + 1) as f64,
        n_permutations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn mean(s: &[f64]) -> f64 {
        s.iter().sum::<f64>() / s.len() as f64
    }

    #[test]
    fn test_bootstrap_mean_ci_matches_t_interval() {
        // For the sample mean, the percentile bootstrap must approximate
        // the classical t-interval.
        let data: Vec<f64> = (0..60)
            .map(|i| 10.0 + ((i * 37) % 17) as f64 * 0.5)
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ci = bootstrap_ci(&data, mean, 5000, 0.95, &mut rng).unwrap();

        let m = mean(&data);
        let sd = (data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 59.0).sqrt();
        let se = sd / (60.0_f64).sqrt();
        assert!((ci.estimate - m).abs() < 1e-12);
        assert!(
            (ci.lower - (m - 2.0 * se)).abs() < 0.5,
            "lower = {}",
            ci.lower
        );
        assert!(
            (ci.upper - (m + 2.0 * se)).abs() < 0.5,
            "upper = {}",
            ci.upper
        );
        assert!((ci.std_error - se).abs() < 0.1);
    }

    #[test]
    fn test_bootstrap_deterministic_with_seed() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut r1 = ChaCha8Rng::seed_from_u64(5);
        let mut r2 = ChaCha8Rng::seed_from_u64(5);
        let c1 = bootstrap_ci(&data, mean, 1000, 0.9, &mut r1).unwrap();
        let c2 = bootstrap_ci(&data, mean, 1000, 0.9, &mut r2).unwrap();
        assert_eq!(c1.lower, c2.lower);
        assert_eq!(c1.upper, c2.upper);
    }

    #[test]
    fn test_permutation_detects_shift_and_respects_null() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let a: Vec<f64> = (0..25).map(|i| ((i * 13) % 7) as f64).collect();
        let b_shifted: Vec<f64> = a.iter().map(|x| x + 4.0).collect();
        let r = permutation_test(
            &a,
            &b_shifted,
            |x, y| mean(x) - mean(y),
            999,
            Alternative::TwoSided,
            &mut rng,
        )
        .unwrap();
        assert!(r.p_value < 0.01, "p = {}", r.p_value);

        // Same distribution: p must be non-significant.
        let b_same: Vec<f64> = (0..25).map(|i| ((i * 5 + 3) % 7) as f64).collect();
        let r = permutation_test(
            &a,
            &b_same,
            |x, y| mean(x) - mean(y),
            999,
            Alternative::TwoSided,
            &mut rng,
        )
        .unwrap();
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn test_permutation_one_sided_consistency() {
        // a clearly below b: Less significant, Greater not.
        let a = [1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.05, 1.15];
        let b = [2.0, 2.2, 1.9, 2.1, 2.3, 1.8, 2.05, 2.15];
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let less = permutation_test(
            &a,
            &b,
            |x, y| mean(x) - mean(y),
            1999,
            Alternative::Less,
            &mut rng,
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let greater = permutation_test(
            &a,
            &b,
            |x, y| mean(x) - mean(y),
            1999,
            Alternative::Greater,
            &mut rng,
        )
        .unwrap();
        assert!(less.p_value < 0.01);
        assert!(greater.p_value > 0.99);
    }

    #[test]
    fn test_errors() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(bootstrap_ci(&[], mean, 1000, 0.95, &mut rng).is_err());
        assert!(bootstrap_ci(&[1.0], mean, 50, 0.95, &mut rng).is_err());
        assert!(bootstrap_ci(&[1.0], mean, 1000, 1.5, &mut rng).is_err());
        assert!(
            permutation_test(
                &[],
                &[1.0],
                |_, _| 0.0,
                999,
                Alternative::TwoSided,
                &mut rng
            )
            .is_err()
        );
    }
}
