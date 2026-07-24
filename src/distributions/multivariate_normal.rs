//! # Multivariate Normal distribution
//!
//! `N(μ, Σ)` over `ℝᵈ`, with flat row-major covariance storage (like the
//! rest of `utils::linalg`). The constructor Cholesky-factorises Σ once —
//! validating positive-definiteness in the process — so `logpdf` is a
//! Mahalanobis form and `sample` a single triangular solve (`μ + L·z`).
//!
//! Not part of the scalar [`Distribution`](crate::Distribution) trait
//! (its support type is a vector); it exposes the same vocabulary as a
//! standalone struct.
//!
//! ```
//! use rs_stats::distributions::multivariate_normal::MultivariateNormal;
//! use rand::SeedableRng;
//!
//! let mvn = MultivariateNormal::new(
//!     vec![1.0, -2.0],
//!     vec![2.0, 0.5,
//!          0.5, 1.5],
//! ).unwrap();
//! let lp = mvn.logpdf(&[1.0, -2.0]).unwrap(); // density at the mean
//! let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
//! let draw = mvn.sample(&mut rng).unwrap();   // Vec<f64> of dim 2
//! assert_eq!(draw.len(), 2);
//! assert!(lp.is_finite());
//! ```

use crate::distributions::normal_distribution::ziggurat_standard_normal;
use crate::error::{StatsError, StatsResult};
use crate::utils::constants::LN_2PI;
use crate::utils::linalg::{cholesky, invert, mahalanobis_sq};

/// Multivariate Normal `N(μ, Σ)` with precomputed Cholesky factor and
/// precision matrix.
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    cov: Vec<f64>,
    dim: usize,
    /// Lower-triangular L with Σ = L·Lᵀ (row-major).
    chol: Vec<f64>,
    /// Σ⁻¹ (row-major).
    cov_inv: Vec<f64>,
    /// −½·(d·ln 2π + ln|Σ|).
    log_norm: f64,
}

impl MultivariateNormal {
    /// Build from a mean vector and a **row-major** `d×d` covariance
    /// matrix. Validates symmetry, finiteness and positive-definiteness
    /// (via Cholesky).
    pub fn new(mean: Vec<f64>, cov: Vec<f64>) -> StatsResult<Self> {
        let dim = mean.len();
        if dim == 0 {
            return Err(StatsError::invalid_input(
                "MultivariateNormal::new: mean must not be empty",
            ));
        }
        if cov.len() != dim * dim {
            return Err(StatsError::invalid_input(format!(
                "MultivariateNormal::new: covariance must be {dim}×{dim} (row-major), got {} elements",
                cov.len()
            )));
        }
        if mean.iter().chain(cov.iter()).any(|v| !v.is_finite()) {
            return Err(StatsError::invalid_input(
                "MultivariateNormal::new: parameters must be finite",
            ));
        }
        for i in 0..dim {
            for j in (i + 1)..dim {
                let (a, b) = (cov[i * dim + j], cov[j * dim + i]);
                if (a - b).abs() > 1e-9 * a.abs().max(b.abs()).max(1.0) {
                    return Err(StatsError::invalid_input(format!(
                        "MultivariateNormal::new: covariance is not symmetric at ({i}, {j})"
                    )));
                }
            }
        }

        let chol = cholesky(&cov, dim)?; // SPD check happens here
        let cov_inv = invert(&cov, dim, 1e-12)?;
        // ln|Σ| = 2·Σᵢ ln Lᵢᵢ — exact and cheap from the factor.
        let log_det: f64 = (0..dim).map(|i| chol[i * dim + i].ln()).sum::<f64>() * 2.0;
        let log_norm = -0.5 * (dim as f64 * LN_2PI + log_det);

        Ok(Self {
            mean,
            cov,
            dim,
            chol,
            cov_inv,
            log_norm,
        })
    }

    /// Dimension `d`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Mean vector μ.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Covariance matrix Σ (row-major).
    pub fn covariance(&self) -> &[f64] {
        &self.cov
    }

    /// Log-density at `x`:
    /// `−½·(d·ln 2π + ln|Σ| + (x−μ)ᵀΣ⁻¹(x−μ))`.
    pub fn logpdf(&self, x: &[f64]) -> StatsResult<f64> {
        if x.len() != self.dim {
            return Err(StatsError::dimension_mismatch(format!(
                "MultivariateNormal::logpdf: expected dim {}, got {}",
                self.dim,
                x.len()
            )));
        }
        let d2 = mahalanobis_sq(x, &self.mean, &self.cov_inv)?;
        Ok(self.log_norm - 0.5 * d2)
    }

    /// Density at `x`.
    pub fn pdf(&self, x: &[f64]) -> StatsResult<f64> {
        Ok(self.logpdf(x)?.exp())
    }

    /// Squared Mahalanobis distance of `x` to the mean — distributed as
    /// χ²(d) under the model (the classic multivariate outlier score).
    pub fn mahalanobis_squared(&self, x: &[f64]) -> StatsResult<f64> {
        mahalanobis_sq(x, &self.mean, &self.cov_inv)
    }

    /// One draw: `μ + L·z` with `z` i.i.d. standard normal (ziggurat).
    pub fn sample(&self, rng: &mut dyn rand::RngCore) -> StatsResult<Vec<f64>> {
        let z: Vec<f64> = (0..self.dim)
            .map(|_| ziggurat_standard_normal(rng))
            .collect();
        let mut out = self.mean.clone();
        for i in 0..self.dim {
            let row = &self.chol[i * self.dim..i * self.dim + i + 1];
            out[i] += row.iter().zip(&z).map(|(l, zi)| l * zi).sum::<f64>();
        }
        Ok(out)
    }

    /// `n` draws, one `Vec` per row.
    pub fn sample_n(&self, rng: &mut dyn rand::RngCore, n: usize) -> StatsResult<Vec<Vec<f64>>> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn mvn3() -> MultivariateNormal {
        MultivariateNormal::new(
            vec![1.0, -2.0, 0.5],
            vec![
                2.0, 0.5, 0.3, //
                0.5, 1.5, -0.2, //
                0.3, -0.2, 1.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_logpdf_matches_scipy() {
        // Reference: scipy.stats.multivariate_normal.logpdf.
        let m = mvn3();
        for (x, expected) in [
            (vec![1.0, -2.0, 0.5], -3.209935797624346),
            (vec![0.0, 0.0, 0.0], -5.306400444088994),
            (vec![3.0, -4.0, 2.0], -6.7548852925738405),
        ] {
            let got = m.logpdf(&x).unwrap();
            assert!((got - expected).abs() < 1e-10, "logpdf({x:?}) = {got}");
        }
    }

    #[test]
    fn test_cholesky_factor_matches_numpy() {
        let m = mvn3();
        let expected = [
            std::f64::consts::SQRT_2, // √cov[0][0] = √2
            0.0,
            0.0, //
            0.35355339059327373,
            1.1726039399558574,
            0.0, //
            0.21213203435596423,
            -0.23452078799117143,
            0.9486832980505138,
        ];
        for (a, b) in m.chol.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sample_moments() {
        let m = mvn3();
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let n = 60_000usize;
        let draws = m.sample_n(&mut rng, n).unwrap();

        // Empirical mean and covariance vs the parameters (~6σ bounds).
        let nf = n as f64;
        let mut mean = [0.0_f64; 3];
        for d in &draws {
            for i in 0..3 {
                mean[i] += d[i];
            }
        }
        for v in mean.iter_mut() {
            *v /= nf;
        }
        for i in 0..3 {
            assert!(
                (mean[i] - m.mean[i]).abs() < 0.04,
                "mean[{i}] = {}",
                mean[i]
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                let c: f64 = draws
                    .iter()
                    .map(|d| (d[i] - mean[i]) * (d[j] - mean[j]))
                    .sum::<f64>()
                    / (nf - 1.0);
                assert!(
                    (c - m.cov[i * 3 + j]).abs() < 0.06,
                    "cov[{i}][{j}] = {c} vs {}",
                    m.cov[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn test_validation_errors() {
        // Not symmetric.
        assert!(MultivariateNormal::new(vec![0.0, 0.0], vec![1.0, 0.5, 0.1, 1.0]).is_err());
        // Not positive-definite.
        assert!(MultivariateNormal::new(vec![0.0, 0.0], vec![1.0, 2.0, 2.0, 1.0]).is_err());
        // Dimension mismatch.
        assert!(MultivariateNormal::new(vec![0.0, 0.0], vec![1.0; 3]).is_err());
        let m = mvn3();
        assert!(m.logpdf(&[0.0, 0.0]).is_err());
    }
}
