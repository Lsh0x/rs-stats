//! # Distribution trait
//!
//! Single unified [`Distribution`] trait covering both continuous and
//! discrete probability distributions, parameterised by an associated
//! support type [`Distribution::X`] — `f64` for continuous, `u64` for
//! discrete. The previous `Distribution` / `DiscreteDistribution` split
//! is gone in v3.0.
//!
//! ## Usage
//!
//! ```
//! use rs_stats::distributions::traits::Distribution;
//! use rs_stats::distributions::normal_distribution::Normal;
//!
//! let n = Normal::new(0.0, 1.0).unwrap();
//! let pdf = n.pdf(0.0).unwrap();
//! assert!((pdf - 0.398_942_280_4).abs() < 1e-8);
//! ```
//!
//! For polymorphism, parameterise on the support type:
//!
//! ```
//! use rs_stats::Distribution;
//! use rs_stats::distributions::normal_distribution::Normal;
//! use rs_stats::distributions::lognormal::LogNormal;
//!
//! fn pick(skewed: bool) -> Box<dyn Distribution<X = f64>> {
//!     if skewed {
//!         Box::new(LogNormal::new(1.0, 0.5).unwrap())
//!     } else {
//!         Box::new(Normal::new(80.0, 10.0).unwrap())
//!     }
//! }
//! ```

use crate::error::{StatsError, StatsResult};

/// Unified probability-distribution interface.
///
/// All methods return [`StatsResult`] to propagate domain errors (e.g.
/// `p ∉ [0, 1]`). The trait is **object-safe** with the support type
/// fixed: `Box<dyn Distribution<X = f64>>` and
/// `Box<dyn Distribution<X = u64>>` both work.
///
/// `fit` is intentionally not part of the trait (it would break object
/// safety); each concrete type exposes `Dist::fit(data)` directly.
pub trait Distribution {
    /// The support type: `f64` for continuous distributions, `u64` for
    /// discrete distributions.
    type X: Copy;

    /// Human-readable distribution name, e.g. `"Normal"`.
    fn name(&self) -> &str;

    /// Number of free parameters — used for AIC / BIC.
    fn num_params(&self) -> usize;

    /// Probability density (continuous) or probability mass (discrete) at `x`.
    fn pdf(&self, x: Self::X) -> StatsResult<f64>;

    /// `ln pdf(x)`. Default: `pdf(x).ln()`. Override for stability when
    /// `pdf` underflows (e.g. extreme tails of LogNormal).
    fn logpdf(&self, x: Self::X) -> StatsResult<f64> {
        self.pdf(x).map(|p| p.ln())
    }

    /// Probability mass — alias for [`pdf`](Self::pdf), kept for natural
    /// reading on discrete distributions and v2.x source-compatibility.
    fn pmf(&self, x: Self::X) -> StatsResult<f64> {
        self.pdf(x)
    }

    /// Log-PMF — alias for [`logpdf`](Self::logpdf).
    fn logpmf(&self, x: Self::X) -> StatsResult<f64> {
        self.logpdf(x)
    }

    /// Cumulative distribution function `F(x) = P(X ≤ x)`.
    fn cdf(&self, x: Self::X) -> StatsResult<f64>;

    /// Survival function `S(x) = P(X > x) = 1 − F(x)`.
    ///
    /// The default computes `1.0 − cdf(x)`, which loses all relative
    /// precision once `S(x) < ~1e-16` (it returns exactly 0). Distributions
    /// with an exact tail form override this — always prefer `sf` over
    /// `1.0 - cdf(x)` for tail probabilities and p-values.
    fn sf(&self, x: Self::X) -> StatsResult<f64> {
        Ok(1.0 - self.cdf(x)?)
    }

    /// Quantile / inverse CDF: smallest `x` such that `F(x) ≥ p`.
    fn inverse_cdf(&self, p: f64) -> StatsResult<Self::X>;

    /// Draw one random sample.
    ///
    /// The default uses inverse-CDF sampling: `inverse_cdf(U)` with
    /// `U ~ Uniform(0, 1)`. Exact but potentially slow for distributions
    /// whose quantile falls back to bisection (Gamma, Beta, F, …).
    ///
    /// Takes `&mut dyn RngCore` so the trait stays object-safe; any
    /// `rand::Rng` (e.g. `rand::thread_rng()`, a seeded `ChaCha8Rng`)
    /// coerces to it.
    fn sample(&self, rng: &mut dyn rand::RngCore) -> StatsResult<Self::X> {
        // Reject u = 0.0 (probability 2⁻⁵³): several quantiles are
        // undefined or −∞ there. u < 1.0 is already guaranteed by gen().
        let u = loop {
            let u: f64 = rand::Rng::r#gen(rng);
            if u > 0.0 {
                break u;
            }
        };
        self.inverse_cdf(u)
    }

    /// Draw `n` random samples into a `Vec`.
    fn sample_n(&self, rng: &mut dyn rand::RngCore, n: usize) -> StatsResult<Vec<Self::X>> {
        (0..n).map(|_| self.sample(rng)).collect()
    }

    /// Mean (expected value) `μ`.
    fn mean(&self) -> f64;

    /// Variance `σ²`.
    fn variance(&self) -> f64;

    /// Standard deviation `σ = √(variance)`.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sum of log-likelihoods `Σ ln pdf(xᵢ)`. Falls back to
    /// [`StatsError`] if any point is out of support.
    fn log_likelihood(&self, data: &[Self::X]) -> StatsResult<f64> {
        let mut ll = 0.0_f64;
        for &x in data {
            ll += self.logpdf(x)?;
        }
        Ok(ll)
    }

    /// Infallible bulk log-likelihood. Out-of-support points contribute
    /// `f64::NEG_INFINITY` (as in scipy). Implementations should override
    /// this when they can express the inner loop without per-point
    /// branching, so LLVM can autovectorise it.
    fn log_likelihood_fast(&self, data: &[Self::X]) -> f64 {
        let mut ll = 0.0_f64;
        for &x in data {
            ll += self.logpdf(x).unwrap_or(f64::NEG_INFINITY);
        }
        ll
    }

    /// Akaike Information Criterion: `2k − 2·ln(L̂)`.
    fn aic(&self, data: &[Self::X]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        Ok(2.0 * self.num_params() as f64 - 2.0 * ll)
    }

    /// Bayesian Information Criterion: `k·ln(n) − 2·ln(L̂)`.
    fn bic(&self, data: &[Self::X]) -> StatsResult<f64> {
        let ll = self.log_likelihood(data)?;
        let n = data.len() as f64;
        Ok(self.num_params() as f64 * n.ln() - 2.0 * ll)
    }
}

// ── Helpers shared across discrete distributions ──────────────────────────────

/// Generic discrete inverse-CDF: smallest `k ≥ 0` such that `cdf(k) ≥ p`.
///
/// Works on any monotone non-decreasing CDF over `u64`. Phase 1
/// exponential-doubling brackets the answer, Phase 2 binary-searches.
/// Discrete distributions whose `inverse_cdf` has no closed form delegate
/// to this helper.
pub(crate) fn discrete_inverse_cdf_search(
    p: f64,
    cdf: impl Fn(u64) -> StatsResult<f64>,
) -> StatsResult<u64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::InvalidInput {
            message: format!("inverse_cdf: p must be in [0, 1], got {p}"),
        });
    }
    if p == 0.0 {
        return Ok(0);
    }
    if p == 1.0 {
        // For unbounded supports the quantile is +∞ (not representable in
        // u64) — and a floating-point CDF that saturates at 1 − ε would send
        // the doubling phase all the way to u64::MAX with O(hi) CDF calls.
        // Bounded distributions (Binomial) special-case p = 1 before
        // delegating here.
        return Err(StatsError::InvalidInput {
            message: "inverse_cdf: p = 1.0 has no finite quantile for unbounded discrete distributions; use p < 1".to_string(),
        });
    }
    let mut hi: u64 = 1;
    while cdf(hi)? < p {
        hi = hi.saturating_mul(2);
        if hi == u64::MAX {
            return Err(StatsError::NumericalError {
                message: "inverse_cdf: quantile exceeds u64::MAX".to_string(),
            });
        }
    }
    let mut lo: u64 = 0;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf(mid)? < p {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// Backwards-compatibility alias. `DiscreteDistribution` was a separate
/// trait in v2.x; in v3.0 every distribution implements the unified
/// [`Distribution`] trait. This alias matches discrete distributions
/// (those whose support is `u64`) and keeps existing imports working.
pub trait DiscreteDistribution: Distribution<X = u64> {}
impl<T: Distribution<X = u64>> DiscreteDistribution for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::normal_distribution::Normal;
    use crate::distributions::poisson_distribution::Poisson;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_sample_normal_matches_distribution() {
        let d = Normal::new(2.0, 1.5).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let xs = d.sample_n(&mut rng, 5000).unwrap();

        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        // ~6σ bounds on the estimators for n = 5000.
        assert!((mean - 2.0).abs() < 0.13, "mean = {mean}");
        assert!((var - 2.25).abs() < 0.30, "var = {var}");

        // KS goodness-of-fit of the sample against its own CDF.
        let ks = crate::distributions::fitting::ks_test(&xs, |x| d.cdf(x).unwrap());
        assert!(ks.p_value > 1e-3, "KS p = {}", ks.p_value);
    }

    #[test]
    fn test_sample_discrete_and_determinism() {
        let d = Poisson::new(3.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let ks = d.sample_n(&mut rng, 4000).unwrap();
        let mean = ks.iter().sum::<u64>() as f64 / ks.len() as f64;
        assert!((mean - 3.0).abs() < 0.12, "mean = {mean}");

        // Same seed ⇒ same stream.
        let mut rng_a = ChaCha8Rng::seed_from_u64(123);
        let mut rng_b = ChaCha8Rng::seed_from_u64(123);
        assert_eq!(
            d.sample_n(&mut rng_a, 20).unwrap(),
            d.sample_n(&mut rng_b, 20).unwrap()
        );
    }

    #[test]
    fn test_gamma_family_samplers_ks() {
        // Every Marsaglia-Tsang-based sampler must pass a KS test against
        // its own CDF (n = 4000, seeded).
        use crate::distributions::beta::Beta;
        use crate::distributions::chi_squared::ChiSquared;
        use crate::distributions::f_distribution::FDistribution;
        use crate::distributions::gamma_distribution::Gamma;
        use crate::distributions::student_t::StudentT;

        fn check<D: Distribution<X = f64>>(name: &str, d: &D, seed: u64) {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let xs = d.sample_n(&mut rng, 4000).unwrap();
            let ks = crate::distributions::fitting::ks_test(&xs, |x| d.cdf(x).unwrap_or(0.0));
            assert!(ks.p_value > 1e-3, "{name}: KS p = {}", ks.p_value);
        }

        check("Gamma(2.5,1.5)", &Gamma::new(2.5, 1.5).unwrap(), 1);
        check("Gamma(0.4,1) boost", &Gamma::new(0.4, 1.0).unwrap(), 2);
        check("Beta(2,5)", &Beta::new(2.0, 5.0).unwrap(), 3);
        check("ChiSquared(4)", &ChiSquared::new(4.0).unwrap(), 4);
        check("StudentT(ν=5)", &StudentT::new(0.0, 1.0, 5.0).unwrap(), 5);
        check(
            "StudentT(ν=1.5) heavy",
            &StudentT::new(0.0, 1.0, 1.5).unwrap(),
            6,
        );
        check("F(5,10)", &FDistribution::new(5.0, 10.0).unwrap(), 7);
    }

    #[test]
    fn test_sf_complements_cdf() {
        let d = Normal::new(0.0, 1.0).unwrap();
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let s = d.cdf(x).unwrap() + d.sf(x).unwrap();
            assert!((s - 1.0).abs() < 1e-12, "cdf+sf = {s} at x = {x}");
        }
        // Deep tail: sf keeps relative precision where 1 − cdf returns 0.
        let tail = d.sf(10.0).unwrap();
        assert!(tail > 0.0 && tail < 1e-20, "sf(10) = {tail}");
    }
}
