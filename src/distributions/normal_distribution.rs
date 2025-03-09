use rand::Rng;
use rand_distr::{Distribution, Normal as RandNormal};
use std::f64::consts::PI;
use crate::prob::erf::erf;
use serde::{Serialize, Deserialize};

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
pub struct NormalConfig {
    /// The mean (μ) of the distribution.
    pub mean: f64,
    /// The standard deviation (σ) of the distribution.
    pub std_dev: f64,
}

impl NormalConfig {
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
    /// assert!(standard_normal.is_some());
    ///
    /// let invalid_config = NormalConfig::new(0.0, -1.0);
    /// assert!(invalid_config.is_none());
    /// ```
    pub fn new(mean: f64, std_dev: f64) -> Option<Self> {
        if std_dev > 0.0 && !mean.is_nan() && !std_dev.is_nan() {
            Some(Self { mean, std_dev })
        } else {
            None
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
/// let pdf = normal_pdf(0.0, 0.0, 1.0);
/// assert!((pdf - 0.3989422804014327).abs() < 1e-10);
///
/// // Normal distribution with mean = 5, std_dev = 2 at x = 5
/// let pdf = normal_pdf(5.0, 5.0, 2.0);
/// assert!((pdf - 0.19947114020071635).abs() < 1e-10);
/// ```
pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    assert!(std_dev > 0.0, "Standard deviation must be positive");
    
    let exponent = -0.5 * ((x - mean) / std_dev).powi(2);
    (1.0 / (std_dev * (2.0 * PI).sqrt())) * exponent.exp()
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
/// let cdf = normal_cdf(0.0, 0.0, 1.0);
/// assert!((cdf - 0.5).abs() < 1e-7);
///
/// // Normal distribution with mean = 5, std_dev = 2 at x = 7
/// let cdf = normal_cdf(7.0, 5.0, 2.0);
/// assert!((cdf - 0.8413447460685429).abs() < 1e-7);
/// ```
pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    assert!(std_dev > 0.0, "Standard deviation must be positive");
    
    // Special case to handle exact value at the mean
    if x == mean {
        return 0.5;
    }

    // Calculate the standardized value z
    let z = (x - mean) / std_dev;
    
    // Use a more numerically stable form of the calculation
    // The sqrt(2) factor is included in the argument to erf
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Calculates the inverse cumulative distribution function (Quantile function) for the normal distribution.
///
/// # Arguments
/// * `p` - Probability value between 0 and 1
/// * `mean` - The mean (μ) of the distribution
/// * `sigma` - The standard deviation (σ) of the distribution
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
/// let p = normal_cdf(x, 0.0, 1.0);
/// let x_back = normal_inverse_cdf(p, 0.0, 1.0);
/// assert!((x - x_back).abs() < 1e-8);
/// ```
pub fn normal_inverse_cdf(p: f64, mean: f64, sigma: f64) -> f64 {
    assert!(p >= 0.0 && p <= 1.0, "Probability must be between 0 and 1");
    
    // Handle edge cases
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    
    // Use a simple and reliable implementation based on the Rational Approximation
    // by Peter J. Acklam
    
    // Convert to standard normal calculation
    let q = if p <= 0.5 {
        p
    } else {
        1.0 - p
    };
    
    // Keep track of whether we need to flip the sign at the end
    let flip_sign = p > 0.5;
    
    // Avoid numerical issues at boundaries
    if q <= 0.0 {
        return if p <= 0.5 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    
    // Coefficients for central region (small |z|)
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    ];
    
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
        1.0
    ];
    
    // Compute rational approximation
    let r = q - 0.5;
    
    let z;
    if q > 0.02425 && q < 0.97575 {
        // Central region
        let r2 = r * r;
        let num = ((((a[0]*r2 + a[1])*r2 + a[2])*r2 + a[3])*r2 + a[4])*r2 + a[5];
        let den = ((((b[0]*r2 + b[1])*r2 + b[2])*r2 + b[3])*r2 + b[4])*r2 + b[5];
        z = r * num / den;
    } else {
        // Tail region
        let s = if r < 0.0 { q } else { 1.0 - q };
        let t = (-2.0 * s.ln()).sqrt();
        
        // Rational approximation for tail
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00
        ];
        
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
            1.0
        ];
        
        let num = ((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5];
        let den = (((d[0]*t + d[1])*t + d[2])*t + d[3])*t + d[4];
        z = if r < 0.0 { -t } else { t } - num / den;
    }
    
    // If p > 0.5, we need to flip the sign of z 
    let final_z = if flip_sign { -z } else { z };
    
    // Convert from standard normal to the specified distribution
    mean + sigma * final_z
}

/// Generates a random sample from the normal distribution.
///
/// # Arguments
/// * `mean` - The mean (μ) of the distribution
/// * `sigma` - The standard deviation (σ) of the distribution
/// * `rng` - A random number generator
///
/// # Returns
/// A random value from the normal distribution
///
/// # Examples
/// ```
/// use rs_stats::distributions::normal_distribution::normal_sample;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let sample = normal_sample(10.0, 2.0, &mut rng);
/// // sample is a random value from Normal(10, 2)
/// ```
pub fn normal_sample<R: Rng>(mean: f64, sigma: f64, rng: &mut R) -> f64 {
    let normal = RandNormal::new(mean, sigma).unwrap();
    normal.sample(rng)
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
        let result = normal_pdf(mean, mean, sigma);
        assert!((result - 0.3989422804014327).abs() < 1e-10);
        
        // Test at one standard deviation away
        let result = normal_pdf(mean + sigma, mean, sigma);
        assert!((result - 0.24197072451914337).abs() < 1e-10);
    }
    
    #[test]
    fn test_normal_pdf_non_standard() {
        let mean = 5.0;
        let sigma = 2.0;
        
        // Test at mean
        let result = normal_pdf(mean, mean, sigma);
        assert!((result - 0.19947114020071635).abs() < 1e-10);
        
        // Test at one standard deviation away
        let result = normal_pdf(mean + sigma, mean, sigma);
        assert!((result - 0.12098536225957168).abs() < 1e-10);
    }
    
    #[test]
    fn test_normal_pdf_symmetry() {
        let mean = 0.0;
        let sigma = 1.0;
        let x = 1.5;
        
        let pdf_plus = normal_pdf(mean + x, mean, sigma);
        let pdf_minus = normal_pdf(mean - x, mean, sigma);
        
        assert!((pdf_plus - pdf_minus).abs() < 1e-10);
    }
    
    #[test]
    fn test_normal_cdf_standard() {
        let mean = 0.0;
        let sigma = 1.0;
        
        // Test at mean
        let result = normal_cdf(mean, mean, sigma);
        assert!((result - 0.5).abs() < 1e-10);
        
        // Test at one standard deviation above mean
        let result = normal_cdf(mean + sigma, mean, sigma);
        assert!((result - 0.8413447460685429).abs() < EPSILON);
        
        // Test at one standard deviation below mean
        let result = normal_cdf(mean - sigma, mean, sigma);
        assert!((result - 0.15865525393145707).abs() < EPSILON);
    }
    
    #[test]
    fn test_normal_cdf_non_standard() {
        let mean = 100.0;
        let sigma = 15.0;
        
        // Test at mean
        let result = normal_cdf(mean, mean, sigma);
        assert!((result - 0.5).abs() < 1e-10);
        
        // Test at one standard deviation above mean
        let result = normal_cdf(mean + sigma, mean, sigma);
        assert!((result - 0.8413447460685429).abs() < EPSILON);
    }
    
    #[test]
    fn test_normal_inverse_cdf() {
        let mean = 0.0;
        let sigma = 1.0;
        
        // Test at median
        let result = normal_inverse_cdf(0.5, mean, sigma);
        assert!((result - mean).abs() < EPSILON);
        
        // Test at one standard deviation above mean
        let result = normal_inverse_cdf(0.8413447460685429, mean, sigma);
        assert!((result - sigma).abs() < EPSILON);
        
        // Test at one standard deviation below mean
        let result = normal_inverse_cdf(0.15865525393145707, mean, sigma);
        assert!((result - (-sigma)).abs() < EPSILON);
    }
    
    #[test]
    fn test_normal_inverse_cdf_non_standard() {
        let mean = 50.0;
        let sigma = 5.0;
        
        // Test at median
        let result = normal_inverse_cdf(0.5, mean, sigma);
        assert!((result - mean).abs() < EPSILON);
        
        // Test at one standard deviation above mean
        let result = normal_inverse_cdf(0.8413447460685429, mean, sigma);
        assert!((result - (mean + sigma)).abs() < EPSILON);
    }
    
    #[test]
    fn test_normal_pdf_standard_normal() {
        // PDF for standard normal at mean should be maximum (approx 0.3989)
        let pdf = (normal_pdf(0.0, 0.0, 1.0) * 1e7).round() / 1e7;
        assert!((pdf - 0.3989423).abs() < EPSILON);
        
        // Test symmetry around mean
        let pdf_plus1 = normal_pdf(1.0, 0.0, 1.0);
        let pdf_minus1 = normal_pdf(-1.0, 0.0, 1.0);
        assert!((pdf_plus1 - pdf_minus1).abs() < EPSILON);
        
        // Test at specific points
        assert!((normal_pdf(1.0, 0.0, 1.0) - 0.2419707).abs() < EPSILON);
        assert!((normal_pdf(2.0, 0.0, 1.0) - 0.0539909).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "Standard deviation must be positive")]
    fn test_normal_pdf_invalid_sigma() {
        normal_pdf(0.0, 0.0, -1.0);
    }

    #[test]
    fn test_normal_cdf_standard_normal() {
        // CDF at mean should be 0.5
        let cdf = (normal_cdf(0.0, 0.0, 1.0) * 1e1).round() / 1e1;
        assert!((cdf - 0.5).abs() < EPSILON);
        
        // Test at specific points
        let cdf = (normal_cdf(1.0, 0.0, 1.0) * 1e7).round() / 1e7;
        assert!((cdf - 0.8413447).abs() < EPSILON);
        
        let cdf = (normal_cdf(-1.0, 0.0, 1.0) * 1e7).round() / 1e7;
        assert!((cdf - 0.1586553).abs() < EPSILON);
        
        let cdf = (normal_cdf(2.0, 0.0, 1.0) * 1e7).round() / 1e7;
        assert!((cdf - 0.9772499).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "Standard deviation must be positive")]
    fn test_normal_cdf_invalid_sigma() {
        normal_cdf(0.0, 0.0, -1.0);
    }

    #[test]
    fn test_normal_inverse_cdf_standard_normal() {
        // Inverse CDF of 0.5 should be the mean (0)
        let x = (normal_inverse_cdf(0.5, 0.0, 1.0) * 1e7).round() / 1e7;
        assert!(x.abs() < EPSILON);
        
        // Test at specific probabilities
        assert!((normal_inverse_cdf(0.8413447, 0.0, 1.0) - 1.0).abs() < 0.01);
        assert!((normal_inverse_cdf(0.1586553, 0.0, 1.0) + 1.0).abs() < 0.01);
    }
}
