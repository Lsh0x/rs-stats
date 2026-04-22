//! # Welford online statistics
//!
//! Numerically stable, single-pass algorithms for incremental computation of
//! mean, variance, standard deviation, and covariance — without storing the
//! full sample.
//!
//! This module provides three estimators :
//!
//! - [`Welford`] — scalar, O(1) per update
//! - [`WelfordVector`] — per-axis multi-dimensional, O(D) per update
//! - [`WelfordCovariance`] — full covariance matrix, O(D²) per update
//!
//! All three support [`merge`](Welford::merge) via Chan's parallel formula
//! (1979) — combining two independent partial estimates into one consolidated
//! estimate, useful for parallel/distributed accumulation.
//!
//! # Why Welford ?
//!
//! The naïve formula `Var(X) = E[X²] − E[X]²` suffers from catastrophic
//! cancellation : both terms become large numbers whose difference is small.
//! Welford accumulates squared deviations from the *running* mean, avoiding
//! the cancellation entirely.
//!
//! # Example
//!
//! ```
//! use rs_stats::prob::welford::Welford;
//!
//! let mut w = Welford::new();
//! for x in [4.0, 7.0, 13.0, 16.0] { w.push(x); }
//! assert_eq!(w.count(), 4);
//! assert!((w.mean() - 10.0).abs() < 1e-12);
//! assert!((w.variance().unwrap() - 30.0).abs() < 1e-12);
//! ```

use crate::error::{StatsError, StatsResult};

// ─────────────────── Scalar Welford ────────────────────────────────────────

/// Online single-pass mean / variance / std-dev estimator for scalar data.
///
/// Numerically stable (Welford 1962). O(1) memory, O(1) per [`push`](Self::push).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Welford {
    n:    u64,
    mean: f64,
    m2:   f64,
}

impl Welford {
    /// Create an empty estimator.
    #[inline]
    pub fn new() -> Self { Self::default() }

    /// Add one observation.
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta  = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2   += delta * delta2;
    }

    /// Remove one observation (subtractive update). Returns an error if the
    /// estimator is empty. The user must guarantee that `x` was previously
    /// added — popping a value that was never inserted produces nonsense
    /// (the algorithm is symmetric, but only valid on the actual sample).
    pub fn pop(&mut self, x: f64) -> StatsResult<()> {
        if self.n == 0 {
            return Err(StatsError::empty_data(
                "prob::welford::Welford::pop: cannot pop from empty estimator",
            ));
        }
        let new_n = self.n - 1;
        if new_n == 0 {
            self.mean = 0.0;
            self.m2   = 0.0;
            self.n    = 0;
            return Ok(());
        }
        let delta  = x - self.mean;
        self.mean  = (self.mean * self.n as f64 - x) / new_n as f64;
        let delta2 = x - self.mean;
        self.m2   -= delta * delta2;
        self.n     = new_n;
        Ok(())
    }

    /// Merge another estimator into this one (Chan parallel formula, 1979).
    /// Combines two independent partial estimates with no information loss.
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 { return; }
        if self.n == 0 { *self = other.clone(); return; }
        let n_total = self.n + other.n;
        let delta   = other.mean - self.mean;
        let new_mean = self.mean + delta * (other.n as f64) / (n_total as f64);
        self.m2 += other.m2 + delta * delta
            * (self.n as f64 * other.n as f64) / (n_total as f64);
        self.mean = new_mean;
        self.n    = n_total;
    }

    /// Number of observations.
    #[inline]
    pub fn count(&self) -> u64 { self.n }

    /// Running mean. Returns 0.0 for an empty estimator.
    #[inline]
    pub fn mean(&self) -> f64 { self.mean }

    /// Sample variance `M2 / (n-1)`. Requires at least 2 observations.
    pub fn variance(&self) -> StatsResult<f64> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "prob::welford::Welford::variance: need at least 2 observations",
            ));
        }
        Ok(self.m2 / (self.n - 1) as f64)
    }

    /// Population variance `M2 / n`. Returns 0.0 for an empty estimator.
    #[inline]
    pub fn population_variance(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.m2 / self.n as f64 }
    }

    /// Sample standard deviation. Requires at least 2 observations.
    pub fn std_dev(&self) -> StatsResult<f64> {
        Ok(self.variance()?.sqrt())
    }

    /// Sum of squared deviations from the mean (`M2`). Useful for serializing
    /// state to disk.
    #[inline]
    pub fn m2(&self) -> f64 { self.m2 }

    /// Reconstruct from raw `(n, mean, m2)` — useful for deserialization.
    pub fn from_raw(n: u64, mean: f64, m2: f64) -> Self {
        Self { n, mean, m2 }
    }
}

// ─────────────────── Vector Welford (per-axis) ─────────────────────────────

/// Online estimator for per-axis mean and variance of D-dimensional data.
///
/// Maintains independent Welford state for each axis — useful for diagonal
/// covariance (Mahalanobis), feature-wise normalisation, etc. O(D) per
/// [`push`](Self::push).
///
/// For full covariance (off-diagonal terms), see [`WelfordCovariance`].
#[derive(Clone, Debug, PartialEq)]
pub struct WelfordVector {
    dim:  usize,
    n:    u64,
    mean: Vec<f64>,
    m2:   Vec<f64>,
}

impl WelfordVector {
    /// Create an empty estimator for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n:    0,
            mean: vec![0.0; dim],
            m2:   vec![0.0; dim],
        }
    }

    /// Add one D-dimensional observation. Returns an error if `x.len() != dim`.
    pub fn push(&mut self, x: &[f64]) -> StatsResult<()> {
        if x.len() != self.dim {
            return Err(StatsError::invalid_input(format!(
                "prob::welford::WelfordVector::push: expected {} dims, got {}",
                self.dim, x.len()
            )));
        }
        self.n += 1;
        let n_inv = 1.0 / self.n as f64;
        for i in 0..self.dim {
            let delta  = x[i] - self.mean[i];
            self.mean[i] += delta * n_inv;
            let delta2 = x[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
        Ok(())
    }

    /// Merge another vector estimator. Both must have the same dimension.
    pub fn merge(&mut self, other: &Self) -> StatsResult<()> {
        if self.dim != other.dim {
            return Err(StatsError::invalid_input(format!(
                "prob::welford::WelfordVector::merge: dim mismatch ({} vs {})",
                self.dim, other.dim
            )));
        }
        if other.n == 0 { return Ok(()); }
        if self.n == 0 { *self = other.clone(); return Ok(()); }
        let n_total = self.n + other.n;
        let n_total_f = n_total as f64;
        let prod_n = (self.n as f64) * (other.n as f64);
        for i in 0..self.dim {
            let delta = other.mean[i] - self.mean[i];
            let new_mean_i = self.mean[i] + delta * (other.n as f64) / n_total_f;
            self.m2[i] += other.m2[i] + delta * delta * prod_n / n_total_f;
            self.mean[i] = new_mean_i;
        }
        self.n = n_total;
        Ok(())
    }

    pub fn count(&self) -> u64 { self.n }
    pub fn dim(&self) -> usize { self.dim }
    pub fn mean(&self) -> &[f64] { &self.mean }
    pub fn m2(&self) -> &[f64] { &self.m2 }

    /// Per-axis sample variance. Requires at least 2 observations.
    pub fn variance(&self) -> StatsResult<Vec<f64>> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "prob::welford::WelfordVector::variance: need at least 2 observations",
            ));
        }
        let denom = (self.n - 1) as f64;
        Ok(self.m2.iter().map(|m| m / denom).collect())
    }

    /// Per-axis sample standard deviation.
    pub fn std_dev(&self) -> StatsResult<Vec<f64>> {
        Ok(self.variance()?.into_iter().map(f64::sqrt).collect())
    }
}

// ─────────────────── Full Covariance Welford ───────────────────────────────

/// Online estimator for the full D×D covariance matrix of D-dimensional data.
///
/// O(D²) per [`push`](Self::push) — heavier than [`WelfordVector`] but captures
/// off-diagonal correlations between axes. Required for full Mahalanobis
/// distance, principal component analysis on a stream, etc.
///
/// Matrix is stored row-major in a flat `Vec<f64>` of length `dim²`.
#[derive(Clone, Debug, PartialEq)]
pub struct WelfordCovariance {
    dim:  usize,
    n:    u64,
    mean: Vec<f64>,
    /// Flattened M2 matrix (dim × dim, row-major).
    m2:   Vec<f64>,
}

impl WelfordCovariance {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n:    0,
            mean: vec![0.0; dim],
            m2:   vec![0.0; dim * dim],
        }
    }

    /// Add one D-dimensional observation. Returns an error if `x.len() != dim`.
    pub fn push(&mut self, x: &[f64]) -> StatsResult<()> {
        if x.len() != self.dim {
            return Err(StatsError::invalid_input(format!(
                "prob::welford::WelfordCovariance::push: expected {} dims, got {}",
                self.dim, x.len()
            )));
        }
        self.n += 1;
        let n_inv = 1.0 / self.n as f64;

        // First compute the pre-update delta and update the mean.
        let mut delta = vec![0.0; self.dim];
        for i in 0..self.dim {
            delta[i] = x[i] - self.mean[i];
            self.mean[i] += delta[i] * n_inv;
        }
        // Then update the M2 matrix using delta · delta_post outer product.
        // Welford's covariance update : M2[i,j] += delta_pre[i] * (x[j] - mean_post[j])
        for i in 0..self.dim {
            for j in 0..self.dim {
                let delta_post_j = x[j] - self.mean[j];
                self.m2[i * self.dim + j] += delta[i] * delta_post_j;
            }
        }
        Ok(())
    }

    /// Merge another covariance estimator (Chan parallel — extended to matrix).
    pub fn merge(&mut self, other: &Self) -> StatsResult<()> {
        if self.dim != other.dim {
            return Err(StatsError::invalid_input(format!(
                "prob::welford::WelfordCovariance::merge: dim mismatch ({} vs {})",
                self.dim, other.dim
            )));
        }
        if other.n == 0 { return Ok(()); }
        if self.n == 0 { *self = other.clone(); return Ok(()); }
        let n_total = self.n + other.n;
        let n_total_f = n_total as f64;
        let prod_n = (self.n as f64) * (other.n as f64);

        let mut delta = vec![0.0; self.dim];
        for i in 0..self.dim {
            delta[i] = other.mean[i] - self.mean[i];
        }

        // Update M2 matrix : M2_AB[i,j] = M2_A[i,j] + M2_B[i,j] + delta[i]*delta[j]*n_A*n_B/n_total
        for i in 0..self.dim {
            for j in 0..self.dim {
                let idx = i * self.dim + j;
                self.m2[idx] += other.m2[idx]
                    + delta[i] * delta[j] * prod_n / n_total_f;
            }
        }
        // Update mean
        for i in 0..self.dim {
            self.mean[i] += delta[i] * (other.n as f64) / n_total_f;
        }
        self.n = n_total;
        Ok(())
    }

    pub fn count(&self) -> u64 { self.n }
    pub fn dim(&self) -> usize { self.dim }
    pub fn mean(&self) -> &[f64] { &self.mean }
    pub fn m2(&self) -> &[f64] { &self.m2 }

    /// Sample covariance matrix `M2 / (n-1)`, row-major flat. Requires
    /// at least 2 observations.
    pub fn covariance(&self) -> StatsResult<Vec<f64>> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "prob::welford::WelfordCovariance::covariance: need at least 2 observations",
            ));
        }
        let denom = (self.n - 1) as f64;
        Ok(self.m2.iter().map(|m| m / denom).collect())
    }
}

// ─────────────────── Tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    #[test]
    fn welford_basic() {
        let mut w = Welford::new();
        for x in [4.0, 7.0, 13.0, 16.0] { w.push(x); }
        assert_eq!(w.count(), 4);
        assert!(approx(w.mean(), 10.0, 1e-12));
        assert!(approx(w.variance().unwrap(), 30.0, 1e-12));
    }

    #[test]
    fn welford_pop_inverts_push() {
        let mut w = Welford::new();
        for x in [4.0, 7.0, 13.0, 16.0] { w.push(x); }
        let before = w.clone();
        w.push(99.0);
        w.pop(99.0).unwrap();
        // Tolerance because pop uses different arithmetic
        assert!(approx(w.mean(), before.mean(), 1e-9));
        assert!(approx(w.m2(), before.m2(), 1e-7));
        assert_eq!(w.count(), before.count());
    }

    #[test]
    fn welford_pop_empty_errors() {
        let mut w = Welford::new();
        assert!(w.pop(1.0).is_err());
    }

    #[test]
    fn welford_merge_chan() {
        let mut a = Welford::new();
        for x in [1.0, 2.0, 3.0] { a.push(x); }
        let mut b = Welford::new();
        for x in [4.0, 5.0, 6.0] { b.push(x); }
        let mut full = Welford::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] { full.push(x); }
        let mut merged = a.clone();
        merged.merge(&b);
        assert!(approx(merged.mean(), full.mean(), 1e-12));
        assert!(approx(merged.variance().unwrap(), full.variance().unwrap(), 1e-12));
    }

    #[test]
    fn welford_merge_with_empty() {
        let mut a = Welford::new();
        a.push(5.0); a.push(10.0);
        let snapshot = a.clone();
        a.merge(&Welford::new());
        assert_eq!(a, snapshot);
        let mut b = Welford::new();
        b.merge(&snapshot);
        assert_eq!(b, snapshot);
    }

    #[test]
    fn welford_population_vs_sample() {
        let mut w = Welford::new();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] { w.push(x); }
        let pop_var = w.population_variance();
        let samp_var = w.variance().unwrap();
        // sample = pop * n / (n-1)
        assert!(approx(samp_var, pop_var * 8.0 / 7.0, 1e-12));
    }

    #[test]
    fn welford_vector_basic() {
        let mut wv = WelfordVector::new(3);
        wv.push(&[1.0, 10.0, 100.0]).unwrap();
        wv.push(&[2.0, 20.0, 200.0]).unwrap();
        wv.push(&[3.0, 30.0, 300.0]).unwrap();
        assert_eq!(wv.mean(), &[2.0, 20.0, 200.0]);
        let var = wv.variance().unwrap();
        assert!(approx(var[0], 1.0, 1e-12));
        assert!(approx(var[1], 100.0, 1e-12));
        assert!(approx(var[2], 10000.0, 1e-12));
    }

    #[test]
    fn welford_vector_merge() {
        let mut a = WelfordVector::new(2);
        a.push(&[1.0, 1.0]).unwrap();
        a.push(&[2.0, 2.0]).unwrap();
        let mut b = WelfordVector::new(2);
        b.push(&[3.0, 3.0]).unwrap();
        b.push(&[4.0, 4.0]).unwrap();
        let mut full = WelfordVector::new(2);
        for v in [[1.0,1.0], [2.0,2.0], [3.0,3.0], [4.0,4.0]] {
            full.push(&v).unwrap();
        }
        let mut merged = a.clone();
        merged.merge(&b).unwrap();
        assert_eq!(merged.count(), 4);
        for i in 0..2 {
            assert!(approx(merged.mean()[i], full.mean()[i], 1e-12));
        }
    }

    #[test]
    fn welford_vector_dim_mismatch_errors() {
        let mut wv = WelfordVector::new(3);
        assert!(wv.push(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn welford_covariance_basic() {
        let mut wc = WelfordCovariance::new(2);
        // x is correlated with y : y = 2x
        wc.push(&[1.0, 2.0]).unwrap();
        wc.push(&[2.0, 4.0]).unwrap();
        wc.push(&[3.0, 6.0]).unwrap();
        wc.push(&[4.0, 8.0]).unwrap();
        let cov = wc.covariance().unwrap();
        // Var(x) = 5/3 ≈ 1.667
        // Var(y) = 4 * 5/3 ≈ 6.667
        // Cov(x,y) = 2 * 5/3 ≈ 3.333 (off-diagonal)
        assert!(approx(cov[0],     5.0 / 3.0,          1e-12));
        assert!(approx(cov[1],     2.0 * 5.0 / 3.0,    1e-12));
        assert!(approx(cov[2],     2.0 * 5.0 / 3.0,    1e-12));
        assert!(approx(cov[3],     4.0 * 5.0 / 3.0,    1e-12));
    }

    #[test]
    fn welford_covariance_merge() {
        let mut a = WelfordCovariance::new(2);
        a.push(&[1.0, 2.0]).unwrap();
        a.push(&[2.0, 4.0]).unwrap();
        let mut b = WelfordCovariance::new(2);
        b.push(&[3.0, 6.0]).unwrap();
        b.push(&[4.0, 8.0]).unwrap();
        let mut full = WelfordCovariance::new(2);
        for v in [[1.0,2.0], [2.0,4.0], [3.0,6.0], [4.0,8.0]] {
            full.push(&v).unwrap();
        }
        let mut merged = a.clone();
        merged.merge(&b).unwrap();
        assert_eq!(merged.count(), 4);
        for i in 0..2 {
            assert!(approx(merged.mean()[i], full.mean()[i], 1e-12));
        }
        let m_cov = merged.covariance().unwrap();
        let f_cov = full.covariance().unwrap();
        for i in 0..4 {
            assert!(approx(m_cov[i], f_cov[i], 1e-9));
        }
    }
}
