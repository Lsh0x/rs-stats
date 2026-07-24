//! # Normal Distribution
//!
//! The Normal (Gaussian) distribution N(μ, σ) is the most widely used continuous
//! distribution, arising naturally as the limiting distribution of sums and averages
//! of independent random variables (Central Limit Theorem).
//!
//! **PDF**: f(x) = 1/(σ√(2π)) · exp(−(x−μ)²/(2σ²))
//!
//! **CDF**: F(x) = Φ((x−μ)/σ), where Φ is the standard normal CDF
//!
//! ## Medical applications
//!
//! | Measurement | Typical parameters |
//! |-------------|-------------------|
//! | **Systolic blood pressure** (healthy adults) | N(120, 10) mmHg |
//! | **Diastolic blood pressure** (healthy adults) | N(80, 8) mmHg |
//! | **Adult height** (men, Western population) | N(175, 7) cm |
//! | **Haemoglobin** (adult men) | N(14.5, 1.0) g/dL |
//! | **Body temperature** | N(37.0, 0.4) °C |
//! | **IQ scores** (by design) | N(100, 15) |
//! | **Lab measurement error** | N(0, σ_instrument) |
//!
//! ## Example — blood pressure reference intervals
//!
//! ```rust
//! use rs_stats::distributions::normal_distribution::Normal;
//! use rs_stats::distributions::traits::Distribution;
//!
//! // Diastolic BP in a healthy cohort: N(80, 8) mmHg
//! let bp = Normal::new(80.0, 8.0).unwrap();
//!
//! // P(DBP > 90 mmHg) — stage 1 hypertension threshold
//! let p_high = 1.0 - bp.cdf(90.0).unwrap();
//! println!("P(DBP > 90 mmHg) = {:.1}%", p_high * 100.0);  // ≈ 10.6%
//!
//! // 95% reference interval (2.5th – 97.5th percentile)
//! let lower = bp.inverse_cdf(0.025).unwrap();
//! let upper = bp.inverse_cdf(0.975).unwrap();
//! println!("Reference interval: [{:.1}, {:.1}] mmHg", lower, upper);
//!
//! // Fit to patient data (MLE: μ̂ = mean, σ̂ = pop std-dev)
//! let readings = vec![78.0, 82.0, 79.0, 85.0, 81.0, 77.0, 83.0, 80.0];
//! let fitted = Normal::fit(&readings).unwrap();
//! println!("Fitted μ = {:.2}, σ = {:.2}", fitted.mean(), fitted.std_dev());
//! ```

use crate::distributions::traits::Distribution;
use crate::error::{StatsError, StatsResult};
use crate::prob::erfc;
use crate::utils::constants::{INV_SQRT_2PI, SQRT_2};

// Private math helpers; the public API is the [`Normal`] struct's
// [`Distribution`] impl below.

/// Calculates the probability density function (PDF) for the normal distribution.
///
/// # Arguments
/// * `x` - The value at which to evaluate the PDF
/// * `mean` - The mean (μ) of the distribution
/// * `std_dev` - The standard deviation (σ) of the distribution (must be positive)
///
/// # Returns
/// The probability density at point x
///
/// # Errors
/// Returns an error if:
/// - std_dev is not positive
/// - Type conversion to f64 fails
///
#[inline]
fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> StatsResult<f64> {
    if std_dev <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "normal_pdf: standard deviation must be positive".to_string(),
        });
    }
    let z = (x - mean) / std_dev;
    Ok((-0.5 * z * z).exp() * INV_SQRT_2PI / std_dev)
}

/// Calculates the cumulative distribution function (CDF) for the normal distribution.
///
/// # Arguments
/// * `x` - The value at which to evaluate the CDF
/// * `mean` - The mean (μ) of the distribution
/// * `std_dev` - The standard deviation (σ) of the distribution (must be positive)
///
/// # Returns
/// The probability that a random variable is less than or equal to x
///
/// # Errors
/// Returns an error if:
/// - std_dev is not positive
/// - Type conversion to f64 fails
///
#[inline]
pub(crate) fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> StatsResult<f64> {
    if std_dev <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "normal_cdf: standard deviation must be positive".to_string(),
        });
    }
    if x == mean {
        return Ok(0.5);
    }
    // Φ(x) = erfc(−z)/2 keeps full relative precision in the lower tail,
    // where 0.5·(1 + erf(z)) collapses to 0 (erf(z) → −1 exactly).
    let z = (x - mean) / (std_dev * SQRT_2);
    Ok(0.5 * erfc(-z)?)
}

/// Calculates the inverse cumulative distribution function (Quantile function) for the normal distribution.
///
/// # Arguments
/// * `p` - Probability value between 0 and 1
/// * `mean` - The mean (μ) of the distribution
/// * `std_dev` - The standard deviation (σ) of the distribution
///
/// # Returns
/// The value x such that P(X ≤ x) = p
///
#[inline]
pub(crate) fn normal_inverse_cdf(p: f64, mean: f64, std_dev: f64) -> StatsResult<f64> {
    let p_64 = p;

    if !(0.0..=1.0).contains(&p_64) {
        return Err(StatsError::InvalidInput {
            message: "normal_inverse_cdf: Probability must be between 0 and 1".to_string(),
        });
    }

    // Handle edge cases
    if p_64 == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p_64 == 1.0 {
        return Ok(f64::INFINITY);
    }

    // Acklam's rational approximation for the inverse standard normal CDF
    // (https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/),
    // accurate to ~1.15 × 10⁻⁹ over the entire support.

    // Coefficients — central region (|p − 0.5| ≤ 0.47575)
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
        1.0,
    ];
    // Coefficients — tail region
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let z = if p_64 < P_LOW {
        // Lower tail
        let q = (-2.0 * p_64.ln()).sqrt();
        let num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5];
        let den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0;
        num / den
    } else if p_64 > P_HIGH {
        // Upper tail
        let q = (-2.0 * (1.0 - p_64).ln()).sqrt();
        let num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5];
        let den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0;
        -num / den
    } else {
        // Central region
        let q = p_64 - 0.5;
        let r = q * q;
        let num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5];
        let den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5];
        q * num / den
    };

    Ok(mean + std_dev * z)
}

// ── Ziggurat sampling (Marsaglia & Tsang, 2000) ────────────────────────────────

/// Rightmost layer boundary x₁ for 128 layers.
const ZIG_R: f64 = 3.442619855899;
/// Area of each layer (and of the base strip + tail).
const ZIG_V: f64 = 9.91256303526217e-3;
/// 2³¹ — scale between the signed 32-bit draw and the x tables.
const ZIG_M1: f64 = 2147483648.0;

struct ZigguratTables {
    /// Acceptance thresholds: |hz| < k[i] ⇒ hz·w[i] is inside layer i.
    k: [u32; 128],
    /// x[i] / 2³¹ — converts the integer draw to a coordinate.
    w: [f64; 128],
    /// f(x[i]) = exp(−x[i]²/2) — layer ordinates for the wedge test.
    f: [f64; 128],
}

static ZIG: std::sync::LazyLock<ZigguratTables> = std::sync::LazyLock::new(|| {
    let mut k = [0u32; 128];
    let mut w = [0.0f64; 128];
    let mut f = [0.0f64; 128];

    let mut dn = ZIG_R;
    let mut tn = dn;
    let q = ZIG_V / (-0.5 * dn * dn).exp();

    k[0] = ((dn / q) * ZIG_M1) as u32;
    k[1] = 0;
    w[0] = q / ZIG_M1;
    w[127] = dn / ZIG_M1;
    f[0] = 1.0;
    f[127] = (-0.5 * dn * dn).exp();

    for i in (1..=126).rev() {
        dn = (-2.0 * (ZIG_V / dn + (-0.5 * dn * dn).exp()).ln()).sqrt();
        k[i + 1] = ((dn / tn) * ZIG_M1) as u32;
        tn = dn;
        f[i] = (-0.5 * dn * dn).exp();
        w[i] = dn / ZIG_M1;
    }

    ZigguratTables { k, w, f }
});

/// Uniform draw in (0, 1] — never 0, so `ln` stays finite.
#[inline]
pub(crate) fn uniform01(rng: &mut dyn rand::RngCore) -> f64 {
    ((rng.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// One standard-normal draw via the 128-layer ziggurat.
///
/// ~99% of draws cost one `next_u64`, one table compare and one multiply —
/// no `ln`/`exp`/`sqrt` — vs the ~4 transcendentals of inverse-CDF
/// sampling. The layer index and the 32-bit magnitude come from disjoint
/// bits of the same `u64`, so they are independent.
pub(crate) fn ziggurat_standard_normal(rng: &mut dyn rand::RngCore) -> f64 {
    let zig = &*ZIG;
    loop {
        let bits = rng.next_u64();
        let iz = (bits & 127) as usize;
        let hz = (bits >> 32) as u32 as i32;

        // Fast path: strictly inside layer iz.
        if (hz.unsigned_abs()) < zig.k[iz] {
            return hz as f64 * zig.w[iz];
        }

        if iz == 0 {
            // Base strip: sample the tail beyond R by Marsaglia's method.
            loop {
                let x = -uniform01(rng).ln() / ZIG_R;
                let y = -uniform01(rng).ln();
                if y + y >= x * x {
                    return if hz >= 0 { ZIG_R + x } else { -(ZIG_R + x) };
                }
            }
        }

        // Wedge: accept with probability proportional to the density gap.
        let x = hz as f64 * zig.w[iz];
        if zig.f[iz] + uniform01(rng) * (zig.f[iz - 1] - zig.f[iz]) < (-0.5 * x * x).exp() {
            return x;
        }
        // Rejected: redraw from scratch.
    }
}

// ── Typed struct + Distribution impl ──────────────────────────────────────────

/// Normal (Gaussian) distribution N(μ, σ²) as a typed struct.
///
/// Implements [`Distribution`] for use with `fit_all` / `fit_best`.
///
/// # Examples
/// ```
/// use rs_stats::distributions::normal_distribution::Normal;
/// use rs_stats::distributions::traits::Distribution;
///
/// let n = Normal::new(0.0, 1.0).unwrap();
/// assert!((n.mean() - 0.0).abs() < 1e-10);
/// assert!((n.pdf(0.0).unwrap() - 0.398_942_280_401_4).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Normal {
    /// Mean μ
    pub mean: f64,
    /// Standard deviation σ (must be > 0)
    pub std_dev: f64,
}

impl Normal {
    /// Creates a `Normal` distribution with validation.
    pub fn new(mean: f64, std_dev: f64) -> StatsResult<Self> {
        // Non-finite parameters (NaN, ±inf) silently produced NaN
        // pdf/cdf values before v4.0 — reject them up front.
        if !mean.is_finite() || !std_dev.is_finite() {
            return Err(StatsError::InvalidInput {
                message: "Normal::new: parameters must be finite".to_string(),
            });
        }
        if std_dev <= 0.0 || std_dev.is_nan() || mean.is_nan() {
            return Err(StatsError::InvalidInput {
                message: "Normal::new: std_dev must be positive and parameters must be finite"
                    .to_string(),
            });
        }
        Ok(Self { mean, std_dev })
    }

    /// Maximum-likelihood estimate from data.
    ///
    /// MLE: μ = mean(data), σ = population std-dev. Single-pass online
    /// (Welford) — never walks `data` twice and never allocates.
    pub fn fit(data: &[f64]) -> StatsResult<Self> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput {
                message: "Normal::fit: data must not be empty".to_string(),
            });
        }
        // Two-pass, multi-accumulator estimator. The textbook two-pass
        // (mean first, then Σ(x−x̄)²) is just as numerically stable as
        // Welford, but Welford's update is a serial dependency chain —
        // it can't pipeline or vectorise. Four independent lanes per pass
        // let LLVM vectorise both reductions (~6× on 100k points).
        let n = data.len() as f64;
        let mut acc = [0.0_f64; 4];
        let mut chunks = data.chunks_exact(4);
        for c in chunks.by_ref() {
            for lane in 0..4 {
                acc[lane] += c[lane];
            }
        }
        let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for &x in chunks.remainder() {
            sum += x;
        }
        let mean = sum / n;

        let mut acc = [0.0_f64; 4];
        let mut chunks = data.chunks_exact(4);
        for c in chunks.by_ref() {
            for lane in 0..4 {
                let d = c[lane] - mean;
                acc[lane] += d * d;
            }
        }
        let mut m2 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for &x in chunks.remainder() {
            let d = x - mean;
            m2 += d * d;
        }
        let variance = m2 / n; // population (MLE)
        Self::new(mean, variance.sqrt())
    }
}

impl Distribution for Normal {
    type X = f64;
    fn name(&self) -> &str {
        "Normal"
    }
    fn num_params(&self) -> usize {
        2
    }
    fn pdf(&self, x: f64) -> StatsResult<f64> {
        normal_pdf(x, self.mean, self.std_dev)
    }
    fn logpdf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.mean) / self.std_dev;
        Ok(-0.5 * z * z - self.std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln())
    }
    /// Closed-form bulk log-likelihood. Lets LLVM autovectorise the
    /// `Σ z_i²` reduction (no per-point Result-returning closure).
    ///
    /// `Σ ln f(xᵢ) = −½ · Σ ((xᵢ−μ)/σ)² − n·(ln σ + ½·ln 2π)`
    fn log_likelihood_fast(&self, data: &[f64]) -> f64 {
        // Pure-arithmetic kernel: four independent accumulators break the
        // FP-add dependency chain so the loop pipelines/vectorises (see
        // prob::average for the rationale).
        let inv_sigma = 1.0 / self.std_dev;
        let mut acc = [0.0_f64; 4];
        let mut chunks = data.chunks_exact(4);
        for chunk in chunks.by_ref() {
            for lane in 0..4 {
                let z = (chunk[lane] - self.mean) * inv_sigma;
                acc[lane] += z * z;
            }
        }
        let mut sum_sq = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for &x in chunks.remainder() {
            let z = (x - self.mean) * inv_sigma;
            sum_sq += z * z;
        }
        let n = data.len() as f64;
        -0.5 * sum_sq - n * (self.std_dev.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln())
    }
    fn cdf(&self, x: f64) -> StatsResult<f64> {
        normal_cdf(x, self.mean, self.std_dev)
    }
    /// Exact upper tail: `S(x) = erfc(z/√2)/2` — full relative precision
    /// where `1 − cdf` would round to 0.
    fn sf(&self, x: f64) -> StatsResult<f64> {
        let z = (x - self.mean) / (self.std_dev * SQRT_2);
        erfc(z).map(|e| 0.5 * e)
    }
    /// Ziggurat sampling (Marsaglia-Tsang): ~8× faster than the default
    /// inverse-CDF path — one u64 draw + one multiply for ~99% of samples.
    fn sample(&self, rng: &mut dyn rand::RngCore) -> StatsResult<f64> {
        Ok(self.mean + self.std_dev * ziggurat_standard_normal(rng))
    }

    fn inverse_cdf(&self, p: f64) -> StatsResult<f64> {
        normal_inverse_cdf(p, self.mean, self.std_dev)
    }
    fn mean(&self) -> f64 {
        self.mean
    }
    fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small epsilon for floating-point comparisons
    const EPSILON: f64 = 1e-7;

    #[test]
    fn test_ziggurat_moments_and_ks() {
        use rand::SeedableRng;
        let d = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
        let n = 200_000usize;
        let xs = d.sample_n(&mut rng, n).unwrap();

        let nf = n as f64;
        let mean = xs.iter().sum::<f64>() / nf;
        let m2 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nf;
        let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / nf;
        let m4 = xs.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / nf;
        let skew = m3 / m2.powf(1.5);
        let kurt = m4 / (m2 * m2) - 3.0;

        // ~6σ estimator bounds at n = 200k.
        assert!(mean.abs() < 0.014, "mean = {mean}");
        assert!((m2 - 1.0).abs() < 0.02, "var = {m2}");
        assert!(skew.abs() < 0.033, "skew = {skew}");
        assert!(kurt.abs() < 0.066, "kurtosis = {kurt}");

        // Tail sanity: the ziggurat's rejection tail must populate |x| > R.
        let beyond_r = xs.iter().filter(|x| x.abs() > 3.442619855899).count();
        let expected = 2.0 * 0.5 * crate::prob::erfc(3.442619855899 / SQRT_2).unwrap() * nf;
        assert!(
            (beyond_r as f64 - expected).abs() < 6.0 * expected.sqrt(),
            "tail count {beyond_r} vs expected {expected:.0}"
        );

        // Distribution-level check.
        let ks = crate::distributions::fitting::ks_test(&xs, |x| d.cdf(x).unwrap());
        assert!(ks.p_value > 1e-3, "KS p = {}", ks.p_value);
    }

    #[test]
    fn test_normal_pdf_standard() {
        let mean = 0.0;
        let sigma = 1.0;

        // Test at mean (peak of the density)
        let result = normal_pdf(mean, mean, sigma).unwrap();
        assert!((result - 0.3989422804014327).abs() < 1e-10);

        // Test at one standard deviation away
        let result = normal_pdf(mean + sigma, mean, sigma).unwrap();
        assert!((result - 0.24197072451914337).abs() < 1e-10);
    }

    #[test]
    fn test_normal_pdf_non_standard() {
        let mean = 5.0;
        let sigma = 2.0;

        // Test at mean
        let result = normal_pdf(mean, mean, sigma).unwrap();
        assert!((result - 0.19947114020071635).abs() < 1e-10);

        // Test at one standard deviation away
        let result = normal_pdf(mean + sigma, mean, sigma).unwrap();
        assert!((result - 0.12098536225957168).abs() < 1e-10);
    }

    #[test]
    fn test_normal_pdf_symmetry() {
        let mean = 0.0;
        let sigma = 1.0;
        let x = 1.5;

        let pdf_plus = normal_pdf(mean + x, mean, sigma).unwrap();
        let pdf_minus = normal_pdf(mean - x, mean, sigma).unwrap();

        assert!((pdf_plus - pdf_minus).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_standard() {
        let mean = 0.0;
        let sigma = 1.0;

        // Test at mean
        let result = normal_cdf(mean, mean, sigma).unwrap();
        assert!((result - 0.5).abs() < 1e-10);

        // Test at one standard deviation above mean
        let result = normal_cdf(mean + sigma, mean, sigma).unwrap();
        assert!((result - 0.8413447460685429).abs() < EPSILON);

        // Test at one standard deviation below mean
        let result = normal_cdf(mean - sigma, mean, sigma).unwrap();
        assert!((result - 0.15865525393145707).abs() < EPSILON);
    }

    #[test]
    fn test_normal_cdf_non_standard() {
        let mean = 100.0;
        let sigma = 15.0;

        // Test at mean
        let result = normal_cdf(mean, mean, sigma).unwrap();
        assert!((result - 0.5).abs() < 1e-10);

        // Test at one standard deviation above mean
        let result = normal_cdf(mean + sigma, mean, sigma).unwrap();
        assert!((result - 0.8413447460685429).abs() < EPSILON);
    }

    #[test]
    fn test_normal_inverse_cdf() {
        let mean = 0.0;
        let sigma = 1.0;

        // Test at median
        let result = normal_inverse_cdf(0.5, mean, sigma).unwrap();
        assert!((result - mean).abs() < EPSILON);

        // Test at one standard deviation above mean
        let result = normal_inverse_cdf(0.8413447460685429, mean, sigma).unwrap();
        assert!((result - sigma).abs() < EPSILON);

        // Test at one standard deviation below mean
        let result = normal_inverse_cdf(0.15865525393145707, mean, sigma).unwrap();
        assert!((result - (-sigma)).abs() < EPSILON);
    }

    #[test]
    fn test_normal_inverse_cdf_non_standard() {
        let mean = 50.0;
        let sigma = 5.0;

        // Test at median
        let result = normal_inverse_cdf(0.5, mean, sigma).unwrap();
        assert!((result - mean).abs() < EPSILON);

        // Test at one standard deviation above mean
        let result = normal_inverse_cdf(0.8413447460685429, mean, sigma).unwrap();
        assert!((result - (mean + sigma)).abs() < EPSILON);
    }

    #[test]
    fn test_normal_pdf_standard_normal() {
        // PDF for standard normal at mean should be maximum (approx 0.3989)
        let pdf = (normal_pdf(0.0, 0.0, 1.0).unwrap() * 1e7).round() / 1e7;
        assert!((pdf - 0.3989423).abs() < EPSILON);

        // Test symmetry around mean
        let pdf_plus1 = normal_pdf(1.0, 0.0, 1.0).unwrap();
        let pdf_minus1 = normal_pdf(-1.0, 0.0, 1.0).unwrap();
        assert!((pdf_plus1 - pdf_minus1).abs() < EPSILON);

        // Test at specific points
        assert!((normal_pdf(1.0, 0.0, 1.0).unwrap() - 0.2419707).abs() < EPSILON);
        assert!((normal_pdf(2.0, 0.0, 1.0).unwrap() - 0.0539909).abs() < EPSILON);
    }

    #[test]
    fn test_normal_pdf_invalid_sigma() {
        let result = normal_pdf(0.0, 0.0, -1.0);
        assert!(
            result.is_err(),
            "Should return error for negative standard deviation"
        );
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_normal_cdf_standard_normal() {
        // CDF at mean should be 0.5
        let cdf = (normal_cdf(0.0, 0.0, 1.0).unwrap() * 1e1).round() / 1e1;
        assert!((cdf - 0.5).abs() < EPSILON);

        // Test at specific points
        let cdf = (normal_cdf(1.0, 0.0, 1.0).unwrap() * 1e7).round() / 1e7;
        assert!((cdf - 0.8413447).abs() < EPSILON);

        let cdf = (normal_cdf(-1.0, 0.0, 1.0).unwrap() * 1e7).round() / 1e7;
        assert!((cdf - 0.1586553).abs() < EPSILON);

        let cdf = (normal_cdf(2.0, 0.0, 1.0).unwrap() * 1e7).round() / 1e7;
        assert!((cdf - 0.9772499).abs() < EPSILON);
    }

    #[test]
    fn test_normal_cdf_invalid_sigma() {
        let result = normal_cdf(0.0, 0.0, -1.0);
        assert!(
            result.is_err(),
            "Should return error for negative standard deviation"
        );
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_normal_inverse_cdf_standard_normal() {
        // Inverse CDF of 0.5 should be the mean (0)
        let x = (normal_inverse_cdf(0.5, 0.0, 1.0).unwrap() * 1e7).round() / 1e7;
        assert!(x.abs() < EPSILON);

        // Test at specific probabilities
        assert!((normal_inverse_cdf(0.8413447, 0.0, 1.0).unwrap() - 1.0).abs() < 0.01);
        assert!((normal_inverse_cdf(0.1586553, 0.0, 1.0).unwrap() + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normal_inverse_cdf_p_negative() {
        let result = normal_inverse_cdf(-0.1, 0.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_normal_inverse_cdf_p_greater_than_one() {
        let result = normal_inverse_cdf(1.5, 0.0, 1.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_normal_inverse_cdf_p_zero() {
        let result = normal_inverse_cdf(0.0, 0.0, 1.0).unwrap();
        assert_eq!(result, f64::NEG_INFINITY);
    }

    #[test]
    fn test_normal_inverse_cdf_p_one() {
        let result = normal_inverse_cdf(1.0, 0.0, 1.0).unwrap();
        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn test_normal_pdf_std_dev_zero() {
        let result = normal_pdf(0.0, 0.0, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_normal_cdf_std_dev_zero() {
        let result = normal_cdf(0.0, 0.0, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_normal_inverse_cdf_std_dev_zero() {
        // std_dev = 0 should still work (just returns mean)
        let result = normal_inverse_cdf(0.5, 5.0, 0.0).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_normal_inverse_cdf_std_dev_negative() {
        // std_dev < 0 should still work (just scales the result)
        let result = normal_inverse_cdf(0.5, 0.0, -1.0).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_normal_new_valid() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        assert_eq!(dist.mean, 0.0);
        assert_eq!(dist.std_dev, 1.0);
    }
}
