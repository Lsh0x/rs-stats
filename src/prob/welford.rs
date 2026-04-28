//! # Welford online statistics
//!
//! Numerically stable, single-pass algorithms for incremental computation of
//! mean, variance, standard deviation, and covariance — without storing the
//! full sample.
//!
//! ## Estimators
//!
//! - [`Welford`] — scalar, O(1) memory and per-update.
//! - [`WelfordVector`] — per-axis multi-dimensional, O(D) per update.
//! - [`WelfordCovariance`] — full covariance matrix, O(D²) per update,
//!   zero per-call allocation thanks to a persistent scratch buffer.
//!
//! ## Why Welford
//!
//! The naïve formula `Var(X) = E[X²] − E[X]²` suffers from catastrophic
//! cancellation: both terms become large numbers whose difference is small.
//! Welford accumulates squared deviations from the *running* mean, avoiding
//! the cancellation entirely. All three estimators in this module support
//! [`merge`](Welford::merge) via Chan's parallel formula (1979) for
//! distributed accumulation.
//!
//! ## Example
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
/// Numerically stable (Welford 1962). O(1) memory, O(1) per
/// [`push`](Self::push). Field layout is preserved on every method —
/// no allocation occurs after construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    /// Construct an empty estimator.
    ///
    /// # Returns
    /// A `Welford` with `count() == 0` and `mean() == 0.0`.
    ///
    /// # Examples
    /// ```
    /// use rs_stats::prob::welford::Welford;
    /// let w = Welford::new();
    /// assert_eq!(w.count(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one observation in O(1).
    ///
    /// # Arguments
    /// * `x` — Sample value.
    ///
    /// # Examples
    /// ```
    /// use rs_stats::prob::welford::Welford;
    /// let mut w = Welford::new();
    /// w.push(1.0);
    /// w.push(2.0);
    /// assert_eq!(w.count(), 2);
    /// ```
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Subtractive update: remove one previously-added observation.
    ///
    /// Uses the numerically stable form `mean -= (x - mean) / new_n` to
    /// avoid the catastrophic cancellation of the textbook
    /// `mean = (mean*n - x) / new_n`. The caller must guarantee that `x`
    /// was previously pushed; popping a value never inserted produces
    /// nonsense (the algorithm is symmetric, but only valid on the
    /// actual sample).
    ///
    /// # Arguments
    /// * `x` — The exact value previously passed to [`push`](Self::push).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if the estimator is empty.
    ///
    /// # Examples
    /// ```
    /// use rs_stats::prob::welford::Welford;
    /// let mut w = Welford::new();
    /// w.push(4.0); w.push(7.0); w.push(99.0);
    /// w.pop(99.0).unwrap();
    /// assert_eq!(w.count(), 2);
    /// ```
    pub fn pop(&mut self, x: f64) -> StatsResult<()> {
        if self.n == 0 {
            return Err(StatsError::empty_data(
                "Welford::pop: cannot pop from empty estimator",
            ));
        }
        let new_n = self.n - 1;
        if new_n == 0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            self.n = 0;
            return Ok(());
        }
        // Stable: mean -= (x - mean) / new_n.
        // The post-update mean satisfies mean_new * new_n = mean_old * n - x,
        // i.e. mean_new = mean_old - (x - mean_old) / new_n. Using this
        // subtractive form avoids `mean*n - x`'s cancellation when the data
        // is centered far from zero.
        let delta = x - self.mean;
        self.mean -= delta / new_n as f64;
        let delta2 = x - self.mean;
        self.m2 -= delta * delta2;
        self.n = new_n;
        Ok(())
    }

    /// Merge another estimator into this one (Chan parallel formula, 1979).
    ///
    /// Combines two independent partial estimates with no information loss.
    /// Useful for parallel / distributed reduction.
    ///
    /// # Arguments
    /// * `other` — Estimator built on a disjoint partition of the sample.
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }
        let n_total = self.n + other.n;
        let delta = other.mean - self.mean;
        let new_mean = self.mean + delta * (other.n as f64) / (n_total as f64);
        self.m2 += other.m2 + delta * delta * (self.n as f64 * other.n as f64) / (n_total as f64);
        self.mean = new_mean;
        self.n = n_total;
    }

    /// Number of observations.
    #[inline]
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Running mean. Returns `0.0` for an empty estimator.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample variance `M2 / (n-1)`.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if `count() < 2` — sample variance
    /// is undefined for fewer than 2 observations.
    pub fn variance(&self) -> StatsResult<f64> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "Welford::variance: need at least 2 observations",
            ));
        }
        Ok(self.m2 / (self.n - 1) as f64)
    }

    /// Population variance `M2 / n`. Returns `0.0` for an empty estimator.
    #[inline]
    pub fn population_variance(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Sample standard deviation.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if `count() < 2`.
    pub fn std_dev(&self) -> StatsResult<f64> {
        Ok(self.variance()?.sqrt())
    }

    /// Sum of squared deviations from the mean (`M2`). Useful for
    /// serializing state to disk.
    #[inline]
    pub fn m2(&self) -> f64 {
        self.m2
    }

    /// Reconstruct from raw `(n, mean, m2)` — useful for deserialization.
    pub fn from_raw(n: u64, mean: f64, m2: f64) -> Self {
        Self { n, mean, m2 }
    }
}

// ─────────────────── Vector Welford (per-axis) ─────────────────────────────

/// Online estimator for per-axis mean and variance of D-dimensional data.
///
/// Maintains independent Welford state for each axis — useful for diagonal
/// covariance, feature-wise normalisation, etc. O(D) per
/// [`push`](Self::push), no per-call allocation.
///
/// For full covariance (off-diagonal terms), see [`WelfordCovariance`].
#[derive(Clone, Debug, PartialEq)]
pub struct WelfordVector {
    dim: usize,
    n: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl WelfordVector {
    /// Construct an empty estimator for `dim`-dimensional vectors.
    ///
    /// # Arguments
    /// * `dim` — Dimensionality of the input vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
        }
    }

    /// Add one D-dimensional observation in O(D), no allocation.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidInput`] if `x.len() != dim`.
    pub fn push(&mut self, x: &[f64]) -> StatsResult<()> {
        if x.len() != self.dim {
            return Err(StatsError::invalid_input(format!(
                "WelfordVector::push: expected {} dims, got {}",
                self.dim,
                x.len()
            )));
        }
        self.n += 1;
        let n_inv = 1.0 / self.n as f64;
        for i in 0..self.dim {
            let delta = x[i] - self.mean[i];
            self.mean[i] += delta * n_inv;
            let delta2 = x[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
        Ok(())
    }

    /// Merge another vector estimator. Both must have the same dimension.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidInput`] on dimension mismatch.
    pub fn merge(&mut self, other: &Self) -> StatsResult<()> {
        if self.dim != other.dim {
            return Err(StatsError::invalid_input(format!(
                "WelfordVector::merge: dim mismatch ({} vs {})",
                self.dim, other.dim
            )));
        }
        if other.n == 0 {
            return Ok(());
        }
        if self.n == 0 {
            *self = other.clone();
            return Ok(());
        }
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

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Configured dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Per-axis running mean.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Per-axis sum of squared deviations.
    pub fn m2(&self) -> &[f64] {
        &self.m2
    }

    /// Per-axis sample variance.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if `count() < 2`.
    pub fn variance(&self) -> StatsResult<Vec<f64>> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "WelfordVector::variance: need at least 2 observations",
            ));
        }
        let denom = (self.n - 1) as f64;
        Ok(self.m2.iter().map(|m| m / denom).collect())
    }

    /// Per-axis sample standard deviation.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if `count() < 2`.
    pub fn std_dev(&self) -> StatsResult<Vec<f64>> {
        Ok(self.variance()?.into_iter().map(f64::sqrt).collect())
    }
}

// ─────────────────── Full Covariance Welford ───────────────────────────────

/// Online estimator for the full D×D covariance matrix of D-dimensional data.
///
/// O(D²) per [`push`](Self::push) — captures off-diagonal correlations
/// (required for full Mahalanobis distance, streaming PCA, etc.). The
/// matrix is stored row-major in a flat `Vec<f64>` of length `dim²` — same
/// layout as [`crate::utils::linalg`] so the two compose without copying.
///
/// **Zero per-call allocation:** a persistent `delta` scratch buffer is
/// kept inside the struct. `push` and `merge` reuse it via `clear() +
/// extend`, never reallocating. This is the headline cost-saving for
/// streaming consumers.
#[derive(Clone, Debug, PartialEq)]
pub struct WelfordCovariance {
    dim: usize,
    n: u64,
    mean: Vec<f64>,
    /// Flattened M2 matrix (dim × dim, row-major).
    m2: Vec<f64>,
    /// Persistent scratch for the per-call `delta` vector. Allocated once
    /// at construction; reused on every `push` / `merge`.
    delta: Vec<f64>,
}

impl WelfordCovariance {
    /// Construct an empty estimator for `dim`-dimensional vectors.
    ///
    /// # Arguments
    /// * `dim` — Dimensionality of the input vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim * dim],
            delta: vec![0.0; dim],
        }
    }

    /// Add one D-dimensional observation in O(D²). No allocation —
    /// the internal `delta` scratch is reused.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidInput`] if `x.len() != dim`.
    pub fn push(&mut self, x: &[f64]) -> StatsResult<()> {
        if x.len() != self.dim {
            return Err(StatsError::invalid_input(format!(
                "WelfordCovariance::push: expected {} dims, got {}",
                self.dim,
                x.len()
            )));
        }
        self.n += 1;
        let n_inv = 1.0 / self.n as f64;

        // Compute pre-update delta and update the mean.
        for i in 0..self.dim {
            self.delta[i] = x[i] - self.mean[i];
            self.mean[i] += self.delta[i] * n_inv;
        }
        // Update the M2 matrix: M2[i,j] += delta_pre[i] * (x[j] - mean_post[j]).
        // Hoist `delta_post_j = x[j] - mean[j]` out of the inner loop so it's
        // computed once per j instead of dim² times.
        for j in 0..self.dim {
            let delta_post_j = x[j] - self.mean[j];
            let row_offset = j;
            for i in 0..self.dim {
                self.m2[i * self.dim + row_offset] += self.delta[i] * delta_post_j;
            }
        }
        Ok(())
    }

    /// Merge another covariance estimator (Chan parallel — extended to matrix).
    ///
    /// Reuses the internal `delta` scratch — no allocation.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidInput`] on dimension mismatch.
    pub fn merge(&mut self, other: &Self) -> StatsResult<()> {
        if self.dim != other.dim {
            return Err(StatsError::invalid_input(format!(
                "WelfordCovariance::merge: dim mismatch ({} vs {})",
                self.dim, other.dim
            )));
        }
        if other.n == 0 {
            return Ok(());
        }
        if self.n == 0 {
            // We can copy other's data into self without realloc since the
            // scratch buffer is sized correctly already.
            self.n = other.n;
            self.mean.copy_from_slice(&other.mean);
            self.m2.copy_from_slice(&other.m2);
            return Ok(());
        }
        let n_total = self.n + other.n;
        let n_total_f = n_total as f64;
        let prod_n = (self.n as f64) * (other.n as f64);

        for i in 0..self.dim {
            self.delta[i] = other.mean[i] - self.mean[i];
        }

        // Update M2 matrix : M2_AB[i,j] = M2_A[i,j] + M2_B[i,j] + delta[i]*delta[j]*n_A*n_B/n_total
        for i in 0..self.dim {
            for j in 0..self.dim {
                let idx = i * self.dim + j;
                self.m2[idx] += other.m2[idx] + self.delta[i] * self.delta[j] * prod_n / n_total_f;
            }
        }
        for i in 0..self.dim {
            self.mean[i] += self.delta[i] * (other.n as f64) / n_total_f;
        }
        self.n = n_total;
        Ok(())
    }

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Configured dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Running mean vector.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Flat row-major M2 matrix (length `dim²`).
    pub fn m2(&self) -> &[f64] {
        &self.m2
    }

    /// Sample covariance matrix `M2 / (n-1)`, row-major flat.
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if `count() < 2`.
    pub fn covariance(&self) -> StatsResult<Vec<f64>> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "WelfordCovariance::covariance: need at least 2 observations",
            ));
        }
        let denom = (self.n - 1) as f64;
        Ok(self.m2.iter().map(|m| m / denom).collect())
    }

    /// Write the sample covariance matrix into a caller-provided buffer.
    /// Zero-allocation variant of [`covariance`](Self::covariance).
    ///
    /// # Errors
    /// Returns [`StatsError::EmptyData`] if `count() < 2`, or
    /// [`StatsError::InvalidInput`] if `out.len() != dim*dim`.
    pub fn covariance_into(&self, out: &mut [f64]) -> StatsResult<()> {
        if self.n < 2 {
            return Err(StatsError::empty_data(
                "WelfordCovariance::covariance_into: need at least 2 observations",
            ));
        }
        if out.len() != self.dim * self.dim {
            return Err(StatsError::invalid_input(format!(
                "WelfordCovariance::covariance_into: out len {} != expected {}",
                out.len(),
                self.dim * self.dim
            )));
        }
        let denom = (self.n - 1) as f64;
        for (i, &m) in self.m2.iter().enumerate() {
            out[i] = m / denom;
        }
        Ok(())
    }
}

// ─────────────────── Tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn welford_basic() {
        let mut w = Welford::new();
        for x in [4.0, 7.0, 13.0, 16.0] {
            w.push(x);
        }
        assert_eq!(w.count(), 4);
        assert!(approx(w.mean(), 10.0, 1e-12));
        assert!(approx(w.variance().unwrap(), 30.0, 1e-12));
    }

    #[test]
    fn welford_pop_inverts_push() {
        let mut w = Welford::new();
        for x in [4.0, 7.0, 13.0, 16.0] {
            w.push(x);
        }
        let before = w.clone();
        w.push(99.0);
        w.pop(99.0).unwrap();
        assert!(approx(w.mean(), before.mean(), 1e-12));
        assert!(approx(w.m2(), before.m2(), 1e-9));
        assert_eq!(w.count(), before.count());
    }

    /// The stable subtractive `pop` formula must hold up under data
    /// centered far from origin (the case where the textbook
    /// `mean = (mean*n - x)/new_n` form catastrophically cancels).
    #[test]
    fn welford_pop_stable_far_from_origin() {
        let mut w = Welford::new();
        // Timestamps in seconds since epoch: 1.7e9, with sub-second variation.
        for i in 0..1000 {
            w.push(1_700_000_000.0 + (i as f64) * 1e-3);
        }
        let before = w.clone();
        w.push(1_700_000_500.123_456);
        w.pop(1_700_000_500.123_456).unwrap();
        // Tight tolerance: stable form keeps relative error ~1e-12 even at
        // this offset (textbook form would drift by ~1e-3).
        assert!(approx(w.mean(), before.mean(), 1e-9));
        assert_eq!(w.count(), before.count());
    }

    #[test]
    fn welford_pop_empty_errors() {
        let mut w = Welford::new();
        assert!(w.pop(1.0).is_err());
    }

    #[test]
    fn welford_pop_to_empty_resets() {
        let mut w = Welford::new();
        w.push(42.0);
        w.pop(42.0).unwrap();
        assert_eq!(w.count(), 0);
        assert_eq!(w.mean(), 0.0);
        assert_eq!(w.m2(), 0.0);
    }

    #[test]
    fn welford_merge_chan() {
        let mut a = Welford::new();
        for x in [1.0, 2.0, 3.0] {
            a.push(x);
        }
        let mut b = Welford::new();
        for x in [4.0, 5.0, 6.0] {
            b.push(x);
        }
        let mut full = Welford::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            full.push(x);
        }
        let mut merged = a.clone();
        merged.merge(&b);
        assert!(approx(merged.mean(), full.mean(), 1e-12));
        assert!(approx(
            merged.variance().unwrap(),
            full.variance().unwrap(),
            1e-12
        ));
    }

    #[test]
    fn welford_merge_with_empty() {
        let mut a = Welford::new();
        a.push(5.0);
        a.push(10.0);
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
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.push(x);
        }
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
        for v in [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]] {
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
        assert!(approx(cov[0], 5.0 / 3.0, 1e-12));
        assert!(approx(cov[1], 2.0 * 5.0 / 3.0, 1e-12));
        assert!(approx(cov[2], 2.0 * 5.0 / 3.0, 1e-12));
        assert!(approx(cov[3], 4.0 * 5.0 / 3.0, 1e-12));
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
        for v in [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]] {
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

    /// covariance_into produces the same values as covariance() and
    /// requires no allocation per call.
    #[test]
    fn welford_covariance_into_matches_covariance() {
        let mut wc = WelfordCovariance::new(3);
        for v in [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]] {
            wc.push(&v).unwrap();
        }
        let owned = wc.covariance().unwrap();
        let mut buf = vec![0.0; 9];
        wc.covariance_into(&mut buf).unwrap();
        for i in 0..9 {
            assert!(approx(owned[i], buf[i], 1e-15));
        }
    }

    #[test]
    fn welford_covariance_into_wrong_size_errors() {
        let mut wc = WelfordCovariance::new(2);
        wc.push(&[1.0, 2.0]).unwrap();
        wc.push(&[3.0, 4.0]).unwrap();
        let mut wrong = vec![0.0; 3];
        assert!(wc.covariance_into(&mut wrong).is_err());
    }
}
