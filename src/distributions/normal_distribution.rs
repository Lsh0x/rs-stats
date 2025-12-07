use num_traits::ToPrimitive;

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use crate::error::{StatsResult, StatsError};
use crate::prob::erf::erf;
use crate::prob::z_score::z_score;


/// Configuration for the Normal distribution.
///
/// # Fields
/// * `mean` - The mean (location parameter)
/// * `std_dev` - The standard deviation (scale parameter, must be positive)
///
/// # Examples
/// ```
/// use rs_stats::distributions::normal_distribution::NormalConfig;
///
/// let config = NormalConfig { mean: 0.0, std_dev: 1.0 };
/// assert!(config.std_dev > 0.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NormalConfig<T> where T: ToPrimitive {
    /// The mean (μ) of the distribution.
    pub mean: T,
    /// The standard deviation (σ) of the distribution.
    pub std_dev: T,
}

impl<T> NormalConfig<T> where T: ToPrimitive {
    /// Creates a new NormalConfig with validation
    ///
    /// # Arguments
    /// * `mean` - The mean of the distribution
    /// * `std_dev` - The standard deviation of the distribution
    ///
    /// # Returns
    /// `Some(NormalConfig)` if parameters are valid, `None` otherwise
    ///
    /// # Examples
    /// ```
    /// use rs_stats::distributions::normal_distribution::NormalConfig;
    ///
    /// let standard_normal = NormalConfig::new(0.0, 1.0);
    /// assert!(standard_normal.is_ok());
    ///
    /// let invalid_config = NormalConfig::new(0.0, -1.0);
    /// assert!(invalid_config.is_err());
    /// ```
    pub fn new(mean: T, std_dev: T) -> StatsResult<Self> {
        let std_dev_64 = std_dev.to_f64().ok_or_else(|| StatsError::ConversionError{
            message: "NormalConfig::new: Failed to convert std_dev to f64".to_string(),
        })?;
        let mean_64 = mean.to_f64().ok_or_else(|| StatsError::ConversionError{
            message: "NormalConfig::new: Failed to convert mean to f64".to_string(),
        })?;
    
        if std_dev_64 > 0.0 && !mean_64.is_nan() && !std_dev_64.is_nan() {
            Ok(Self { mean, std_dev })
        } else {
            Err(StatsError::InvalidInput {
                message: "NormalConfig::new: std_dev must be positive".to_string(),
            })
        }
    }
}

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
/// # Panics
/// Panics if std_dev is not positive.
///
/// # Examples
/// ```
/// use rs_stats::distributions::normal_distribution::normal_pdf;
///
/// // Standard normal distribution at x = 0
/// let pdf = normal_pdf(0.0, 0.0, 1.0).unwrap();
/// assert!((pdf - 0.3989422804014327).abs() < 1e-10);
///
/// // Normal distribution with mean = 5, std_dev = 2 at x = 5
/// let pdf = normal_pdf(5.0, 5.0, 2.0).unwrap();
/// assert!((pdf - 0.19947114020071635).abs() < 1e-10);
/// ```
pub fn normal_pdf<T>(x: T, mean: f64, std_dev: f64) -> StatsResult<f64> where T: ToPrimitive {
    if std_dev <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "normal_pdf: Standard deviation must be positive".to_string(),
        });
    }
    
    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "normal_pdf: Failed to convert x to f64".to_string(),
    })?;

    let exponent = -0.5 * ((x_64 - mean) / std_dev).powi(2);
    Ok((1.0 / (std_dev * (2.0 * PI).sqrt())) * exponent.exp())
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
/// # Panics
/// Panics if std_dev is not positive.
///
/// # Examples
/// ```
/// use rs_stats::distributions::normal_distribution::normal_cdf;
///
/// // Standard normal distribution at x = 0
/// let cdf = normal_cdf(0.0, 0.0, 1.0).unwrap();
/// assert!((cdf - 0.5).abs() < 1e-7);
///
/// // Normal distribution with mean = 5, std_dev = 2 at x = 7
/// let cdf = normal_cdf(7.0, 5.0, 2.0).unwrap();
/// assert!((cdf - 0.8413447460685429).abs() < 1e-7);
/// ```
pub fn normal_cdf<T>(x: T, mean: f64, std_dev: f64) -> StatsResult<f64> where T: ToPrimitive {
    if std_dev <= 0.0 {
        return Err(StatsError::InvalidInput {
            message: "normal_cdf: Standard deviation must be positive".to_string(),
        });
    }

    let x_64 = x.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "normal_cdf: Failed to convert x to f64".to_string(),
    })?;

    // Special case to handle exact value at the mean
    if x_64 == mean {
        return Ok(0.5);
    }

    let z = z_score(x_64, mean, std_dev)?;

    Ok(0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2)?))
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
/// # Examples
/// ```
/// use rs_stats::distributions::normal_distribution::{normal_cdf, normal_inverse_cdf};
///
/// // Check that inverse_cdf is the inverse of cdf
/// let x = 0.5;
/// let p = normal_cdf(x, 0.0, 1.0).unwrap();
/// let x_back = normal_inverse_cdf(p, 0.0, 1.0).unwrap();
/// assert!((x - x_back).abs() < 1e-8);
/// ```
pub fn normal_inverse_cdf<T>(p: T, mean: f64, std_dev: f64) -> StatsResult<f64> where T: ToPrimitive{
    let p_64 = p.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "normal_inverse_cdf: Failed to convert p to f64".to_string(),
    })?;

    if p_64 < 0.0 || p_64 > 1.0 {
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

    // Use a simple and reliable implementation based on the Rational Approximation
    // by Peter J. Acklam

    // Convert to standard normal calculation
    let q = if p_64 <= 0.5 { p_64 } else { 1.0 - p_64 };

    // Avoid numerical issues at boundaries
    if q <= 0.0 {
        return if p_64 <= 0.5 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(f64::INFINITY)
        };
    }

    // Coefficients for central region (small |z|)
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

    // Compute rational approximation
    let r = q - 0.5;

    let z = if q > 0.02425 && q < 0.97575 {
        // Central region
        let r2 = r * r;
        let num = ((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5];
        let den = ((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + b[5];
        r * num / den
    } else {
        // Tail region
        let s = if r < 0.0 { q } else { 1.0 - q };
        let t = (-2.0 * s.ln()).sqrt();

        // Rational approximation for tail
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
            1.0,
        ];

        let num = ((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5];
        let den = (((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + d[4];
        if r < 0.0 {
            -t - num / den
        } else {
            t - num / den
        }
    };

    // If p > 0.5, we need to flip the sign of z
    let final_z = if p_64 > 0.5 { z * -1.0 } else { z };

    let result = mean + std_dev * final_z;
    // Convert from standard normal to the specified distribution
    Ok(result)
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
        assert!(result.is_err(), "Should return error for negative standard deviation");
        assert!(matches!(result.unwrap_err(), StatsError::InvalidInput { .. }));
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
        assert!(result.is_err(), "Should return error for negative standard deviation");
        assert!(matches!(result.unwrap_err(), StatsError::InvalidInput { .. }));
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
}
