//! # Linear algebra helpers
//!
//! Small dense linear algebra routines used by statistical primitives —
//! matrix inversion with optional Tikhonov (ridge) regularization, and
//! Mahalanobis / weighted-L2 distances built on top.
//!
//! Matrices are represented as **row-major flat `Vec<f64>`** of length `n²`.
//! This is the same layout used by
//! [`WelfordCovariance::covariance`](crate::prob::welford::WelfordCovariance::covariance)
//! so the two compose without copying.
//!
//! All public entry points have a *zero-allocation* `_into` variant that
//! takes a caller-provided scratch buffer — important for hot paths
//! (e.g. nearest-neighbour search) where allocating a `Vec` per call
//! dominates wall-clock time.

use crate::error::{StatsError, StatsResult};

/// In-place Gauss-Jordan inversion with partial pivoting.
///
/// Inverts a square `dim × dim` matrix. Owns its augmented buffer (one
/// allocation of `dim * 2 * dim` floats); the inverse is returned in a
/// fresh `Vec`.
///
/// # Arguments
/// * `matrix` — Row-major flat `Vec<f64>` of length `dim²`.
/// * `dim` — Side length of the square matrix.
/// * `eps` — Pivot tolerance: a pivot smaller than this aborts as singular.
///
/// # Returns
/// Row-major flat `Vec<f64>` of the inverse, length `dim²`.
///
/// # Errors
/// * [`StatsError::InvalidInput`] — `matrix.len() != dim*dim`.
/// * [`StatsError::NumericalError`] — matrix is singular within `eps`.
///
/// # Examples
/// ```
/// use rs_stats::utils::linalg::invert;
/// // 2×2: [[4,7],[2,6]] → det 10, inverse (1/10)·[[6,-7],[-2,4]]
/// let m = vec![4.0, 7.0, 2.0, 6.0];
/// let inv = invert(&m, 2, 1e-9).unwrap();
/// assert!((inv[0] - 0.6).abs() < 1e-12);
/// ```
pub fn invert(matrix: &[f64], dim: usize, eps: f64) -> StatsResult<Vec<f64>> {
    if matrix.len() != dim * dim {
        return Err(StatsError::invalid_input(format!(
            "linalg::invert: expected {} elements for {}×{} matrix, got {}",
            dim * dim,
            dim,
            dim,
            matrix.len()
        )));
    }
    // Build [A | I] augmented matrix of size dim × 2*dim.
    let w = 2 * dim;
    let mut aug = vec![0.0; dim * w];
    for r in 0..dim {
        for c in 0..dim {
            aug[r * w + c] = matrix[r * dim + c];
        }
        aug[r * w + dim + r] = 1.0;
    }

    // Forward elimination with partial pivoting.
    for col in 0..dim {
        // Find pivot row.
        let mut pivot_row = col;
        let mut pivot_val = aug[col * w + col].abs();
        for r in (col + 1)..dim {
            let v = aug[r * w + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < eps {
            return Err(StatsError::numerical_error(format!(
                "linalg::invert: matrix is singular (pivot {} < eps {})",
                pivot_val, eps
            )));
        }
        if pivot_row != col {
            for c in 0..w {
                aug.swap(col * w + c, pivot_row * w + c);
            }
        }
        let inv_pivot = 1.0 / aug[col * w + col];
        for c in 0..w {
            aug[col * w + c] *= inv_pivot;
        }
        for r in 0..dim {
            if r == col {
                continue;
            }
            let factor = aug[r * w + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..w {
                aug[r * w + c] -= factor * aug[col * w + c];
            }
        }
    }

    // Extract inverse (right half of augmented matrix).
    let mut inv = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            inv[r * dim + c] = aug[r * w + dim + c];
        }
    }
    Ok(inv)
}

/// Invert with Tikhonov (ridge) regularization.
///
/// Adds `λ·I` to the diagonal before inverting, where
/// `λ = trace(M) / (dim · ridge_factor)` (auto-scaled to the matrix
/// magnitude). Standard practice for covariance matrices that may be
/// near-singular due to small samples, quantisation, or collinearity.
///
/// # Arguments
/// * `matrix` — Row-major flat `Vec<f64>` of length `dim²`.
/// * `dim` — Side length.
/// * `ridge_factor` — Regularisation knob:
///   - **Large (100+)** → gentle, near the pure inverse.
///   - **Small (1)** → heavy, inverse tends toward `(1/λ)·I`.
///
/// # Returns
/// Row-major flat `Vec<f64>` of the regularised inverse.
///
/// # Errors
/// * [`StatsError::InvalidInput`] — bad length.
/// * [`StatsError::NumericalError`] — singular even after regularisation
///   (extremely unusual; suggests `ridge_factor` too large for the matrix).
///
/// # Examples
/// ```
/// use rs_stats::utils::linalg::invert_with_ridge;
/// // Singular rank-1 matrix; pure invert would fail.
/// let m = vec![1.0, 2.0, 2.0, 4.0];
/// let inv = invert_with_ridge(&m, 2, 10.0);
/// assert!(inv.is_ok());
/// ```
pub fn invert_with_ridge(matrix: &[f64], dim: usize, ridge_factor: f64) -> StatsResult<Vec<f64>> {
    if matrix.len() != dim * dim {
        return Err(StatsError::invalid_input(format!(
            "linalg::invert_with_ridge: expected {} elements, got {}",
            dim * dim,
            matrix.len()
        )));
    }
    let mut trace = 0.0;
    for i in 0..dim {
        trace += matrix[i * dim + i];
    }
    let lambda = (trace / dim as f64 / ridge_factor.max(1e-9)).max(1e-12);
    let mut reg = matrix.to_vec();
    for i in 0..dim {
        reg[i * dim + i] += lambda;
    }
    invert(&reg, dim, 1e-9)
}

/// Mahalanobis-style squared distance `(x-μ)ᵀ · M · (x-μ)`.
///
/// Allocates two `Vec<f64>` of length `dim` per call. **Not suitable for
/// per-vector hot paths** — use [`mahalanobis_sq_into`] instead, which
/// writes through a caller-provided scratch buffer and never allocates.
///
/// # Arguments
/// * `x` — Query vector.
/// * `mean` — Reference mean / centroid.
/// * `m_inv` — Row-major flat dim² matrix (typically a precomputed
///   inverse covariance).
///
/// # Returns
/// The squared distance (always ≥ 0 for positive-semi-definite `m_inv`).
///
/// # Errors
/// [`StatsError::InvalidInput`] on length mismatch.
///
/// # Examples
/// ```
/// use rs_stats::utils::linalg::mahalanobis_sq;
/// let x = vec![1.0, 2.0, 3.0];
/// let mean = vec![0.0, 0.0, 0.0];
/// let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// // Mahalanobis with identity reduces to squared L2.
/// let d = mahalanobis_sq(&x, &mean, &identity).unwrap();
/// assert!((d - 14.0).abs() < 1e-12);
/// ```
pub fn mahalanobis_sq(x: &[f64], mean: &[f64], m_inv: &[f64]) -> StatsResult<f64> {
    let dim = x.len();
    let mut d = vec![0.0; dim];
    let mut md = vec![0.0; dim];
    mahalanobis_sq_into(x, mean, m_inv, &mut d, &mut md)
}

/// Zero-allocation variant of [`mahalanobis_sq`].
///
/// Writes intermediate values through caller-provided scratch buffers.
/// Hoist the buffers out of the inner loop in any per-row scoring pass.
///
/// # Arguments
/// * `x`, `mean`, `m_inv` — see [`mahalanobis_sq`].
/// * `scratch_diff` — Output for `x − mean`. Length must be `x.len()`.
/// * `scratch_md` — Output for `m_inv · (x − mean)`. Length must be `x.len()`.
///
/// # Returns
/// The squared distance.
///
/// # Errors
/// [`StatsError::InvalidInput`] on length mismatch in any argument.
///
/// # Examples
/// ```
/// use rs_stats::utils::linalg::mahalanobis_sq_into;
/// // Hot-path pattern: hoist scratch out of the loop.
/// let mean = vec![0.0, 0.0];
/// let m = vec![1.0, 0.0, 0.0, 4.0];
/// let mut diff = vec![0.0; 2];
/// let mut md = vec![0.0; 2];
/// for x in [[1.0, 1.0], [2.0, 0.5], [-1.0, 3.0]] {
///     let d = mahalanobis_sq_into(&x, &mean, &m, &mut diff, &mut md).unwrap();
///     assert!(d >= 0.0);
/// }
/// ```
pub fn mahalanobis_sq_into(
    x: &[f64],
    mean: &[f64],
    m_inv: &[f64],
    scratch_diff: &mut [f64],
    scratch_md: &mut [f64],
) -> StatsResult<f64> {
    let dim = x.len();
    if mean.len() != dim {
        return Err(StatsError::invalid_input(format!(
            "linalg::mahalanobis_sq_into: mean dim {} != x dim {}",
            mean.len(),
            dim
        )));
    }
    if m_inv.len() != dim * dim {
        return Err(StatsError::invalid_input(format!(
            "linalg::mahalanobis_sq_into: m_inv dim {} != expected {}",
            m_inv.len(),
            dim * dim
        )));
    }
    if scratch_diff.len() != dim || scratch_md.len() != dim {
        return Err(StatsError::invalid_input(format!(
            "linalg::mahalanobis_sq_into: scratch buffers must have len {}",
            dim
        )));
    }
    for i in 0..dim {
        scratch_diff[i] = x[i] - mean[i];
    }
    for r in 0..dim {
        let mut s = 0.0;
        let row = r * dim;
        for c in 0..dim {
            s += m_inv[row + c] * scratch_diff[c];
        }
        scratch_md[r] = s;
    }
    let mut score = 0.0;
    for i in 0..dim {
        score += scratch_diff[i] * scratch_md[i];
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn invert_identity_is_identity() {
        let i = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert(&i, 3, 1e-9).unwrap();
        for (a, b) in i.iter().zip(inv.iter()) {
            assert!(approx(*a, *b, 1e-12));
        }
    }

    #[test]
    fn invert_2x2() {
        // [[4, 7], [2, 6]] → det = 24 - 14 = 10
        // inverse = (1/10) * [[6, -7], [-2, 4]]
        let a = vec![4.0, 7.0, 2.0, 6.0];
        let inv = invert(&a, 2, 1e-9).unwrap();
        assert!(approx(inv[0], 0.6, 1e-12));
        assert!(approx(inv[1], -0.7, 1e-12));
        assert!(approx(inv[2], -0.2, 1e-12));
        assert!(approx(inv[3], 0.4, 1e-12));
    }

    #[test]
    fn invert_singular_errors() {
        // Rank 1 matrix.
        let a = vec![1.0, 2.0, 2.0, 4.0];
        assert!(invert(&a, 2, 1e-9).is_err());
    }

    #[test]
    fn invert_a_times_inv_is_identity() {
        let a = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 4.0];
        let inv = invert(&a, 3, 1e-9).unwrap();
        // Compute A · A^-1 and check it's I.
        let mut prod = vec![0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[r * 3 + k] * inv[k * 3 + c];
                }
                prod[r * 3 + c] = s;
            }
        }
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (p, i) in prod.iter().zip(identity.iter()) {
            assert!(approx(*p, *i, 1e-9));
        }
    }

    #[test]
    fn invert_wrong_size_errors() {
        let a = vec![1.0; 5]; // 5 elements, but dim=2 → expected 4.
        assert!(invert(&a, 2, 1e-9).is_err());
    }

    #[test]
    fn invert_with_ridge_handles_singular() {
        let a = vec![1.0, 2.0, 2.0, 4.0]; // singular
        let inv = invert_with_ridge(&a, 2, 10.0);
        assert!(inv.is_ok());
    }

    #[test]
    fn mahalanobis_identity_is_l2() {
        let x = vec![1.0, 2.0, 3.0];
        let mean = vec![0.0, 0.0, 0.0];
        let i = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let d = mahalanobis_sq(&x, &mean, &i).unwrap();
        assert!(approx(d, 1.0 + 4.0 + 9.0, 1e-12));
    }

    #[test]
    fn mahalanobis_diag_weighted() {
        let x = vec![2.0, 2.0];
        let mean = vec![0.0, 0.0];
        // M = diag(1, 4) → d² = 1·4 + 4·4 = 20
        let m = vec![1.0, 0.0, 0.0, 4.0];
        let d = mahalanobis_sq(&x, &mean, &m).unwrap();
        assert!(approx(d, 20.0, 1e-12));
    }

    #[test]
    fn mahalanobis_sq_into_matches_owning_variant() {
        let x = vec![1.0, 2.0, 3.0];
        let mean = vec![0.5, 0.5, 0.5];
        let m = vec![2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5];
        let owned = mahalanobis_sq(&x, &mean, &m).unwrap();
        let mut diff = vec![0.0; 3];
        let mut md = vec![0.0; 3];
        let scratched = mahalanobis_sq_into(&x, &mean, &m, &mut diff, &mut md).unwrap();
        assert!(approx(owned, scratched, 1e-15));
    }

    #[test]
    fn mahalanobis_sq_into_wrong_scratch_errors() {
        let x = vec![1.0, 2.0];
        let mean = vec![0.0, 0.0];
        let m = vec![1.0, 0.0, 0.0, 1.0];
        let mut diff = vec![0.0; 1]; // wrong size
        let mut md = vec![0.0; 2];
        assert!(mahalanobis_sq_into(&x, &mean, &m, &mut diff, &mut md).is_err());
    }
}
