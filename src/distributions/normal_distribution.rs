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
use crate::prob::erf;
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
    let z = (x - mean) / (std_dev * SQRT_2);
    Ok(0.5 * (1.0 + erf(z)?))
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
pub struct Normal {
    /// Mean μ
    pub mean: f64,
    /// Standard deviation σ (must be > 0)
    pub std_dev: f64,
}

impl Normal {
    /// Creates a `Normal` distribution with validation.
    pub fn new(mean: f64, std_dev: f64) -> StatsResult<Self> {
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
        let mut count = 0.0_f64;
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for &x in data {
            count += 1.0;
            let delta = x - mean;
            mean += delta / count;
            m2 += delta * (x - mean);
        }
        let variance = m2 / count; // population (MLE)
        Self::new(mean, variance.sqrt())
    }
}

impl Distribution for Normal {
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
    fn cdf(&self, x: f64) -> StatsResult<f64> {
        normal_cdf(x, self.mean, self.std_dev)
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
