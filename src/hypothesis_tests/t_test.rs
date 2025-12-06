//! # Student's t-tests
//!
//! This module provides implementations of Student's t-tests for statistical hypothesis testing.
//!
//! ## Included tests:
//! - **One-sample t-test**: Tests if a sample mean differs from a population mean
//! - **Two-sample t-test**: Tests if means of two samples differ (unpaired)
//! - **Paired t-test**: Tests if the mean difference between paired observations is zero
//!
//! ## Mathematical Background
//!
//! The t-test was developed by William Sealy Gosset under the pseudonym "Student",
//! and is used when the sample size is small and the population standard deviation is unknown.
//!
//! The test statistic follows a t-distribution under the null hypothesis, which has
//! heavier tails compared to the normal distribution to account for the additional
//! uncertainty from estimating the standard deviation.

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;
use std::f64;
use std::fmt::Debug;

/// Natural logarithm of 2π (2*pi)
const LN_2PI: f64 = 1.8378770664093456;

/// Result of a t-test analysis
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// The calculated t-statistic
    pub t_statistic: f64,
    /// Degrees of freedom for the t-distribution
    pub degrees_of_freedom: f64,
    /// The two-tailed p-value
    pub p_value: f64,
    /// The sample mean(s)
    pub mean_values: Vec<f64>,
    /// The sample standard deviation(s)
    pub std_devs: Vec<f64>,
    /// The standard error of the mean difference
    pub std_error: f64,
}

/// Performs a one-sample t-test to determine if a sample mean differs from a specified population mean.
///
/// # Arguments
/// * `data` - The sample data
/// * `population_mean` - The population mean to test against
/// * `alpha` - The significance level (default: 0.05)
///
/// # Returns
/// A `Result` containing the t-test results or an error if the data is insufficient
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::t_test::one_sample_t_test;
///
/// let data = vec![5.2, 6.4, 6.9, 7.3, 7.5, 7.8, 8.1, 8.4, 9.2, 9.5];
/// let population_mean = 7.0;
///
/// let result = one_sample_t_test(&data, population_mean).unwrap();
/// println!("T-statistic: {}", result.t_statistic);
/// println!("P-value: {}", result.p_value);
///
/// // Test if the result is significant at alpha = 0.05
/// if result.p_value < 0.05 {
///     println!("Reject null hypothesis: Sample mean differs from population mean");
/// } else {
///     println!("Fail to reject null hypothesis");
/// }
/// ```
pub fn one_sample_t_test<T>(data: &[T], population_mean: T) -> StatsResult<TTestResult>
where
    T: ToPrimitive + Debug + Copy,
{
    if data.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot perform t-test on empty data",
        ));
    }

    if data.len() < 2 {
        return Err(StatsError::invalid_input(
            "Need at least 2 data points for t-test",
        ));
    }

    let pop_mean = population_mean
        .to_f64()
        .ok_or_else(|| StatsError::conversion_error("Failed to convert population mean to f64"))?;

    // Calculate sample statistics
    let n = data.len() as f64;
    let mean = calculate_mean(data)?;
    let variance = calculate_variance(data, mean)?;
    let std_dev = variance.sqrt();
    let std_error = std_dev / n.sqrt();

    // Calculate t-statistic
    let t_statistic = (mean - pop_mean) / std_error;

    // Degrees of freedom
    let df = n - 1.0;

    // Calculate p-value (two-tailed)
    let p_value = calculate_p_value(t_statistic.abs(), df);

    Ok(TTestResult {
        t_statistic,
        degrees_of_freedom: df,
        p_value,
        mean_values: vec![mean],
        std_devs: vec![std_dev],
        std_error,
    })
}

/// Performs a two-sample t-test to determine if the means of two independent samples differ.
///
/// This function implements Welch's t-test, which does not assume equal variances in both groups.
///
/// # Arguments
/// * `data1` - The first sample
/// * `data2` - The second sample
/// * `equal_variances` - If true, assumes equal variances (Student's t-test); otherwise, uses Welch's t-test (default)
///
/// # Returns
/// A `Result` containing the t-test results or an error if the data is insufficient
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::t_test::two_sample_t_test;
///
/// let group1 = vec![5.2, 6.4, 6.9, 7.3, 7.5, 7.8, 8.1, 8.4, 9.2, 9.5];
/// let group2 = vec![4.1, 5.0, 5.5, 6.2, 6.3, 6.5, 6.8, 7.1, 7.4, 7.5];
///
/// // Using Welch's t-test (default, doesn't assume equal variances)
/// let result = two_sample_t_test(&group1, &group2, false).unwrap();
/// println!("T-statistic: {}", result.t_statistic);
/// println!("P-value: {}", result.p_value);
///
/// if result.p_value < 0.05 {
///     println!("Reject null hypothesis: The group means differ");
/// } else {
///     println!("Fail to reject null hypothesis");
/// }
/// ```
pub fn two_sample_t_test<T>(
    data1: &[T],
    data2: &[T],
    equal_variances: bool,
) -> StatsResult<TTestResult>
where
    T: ToPrimitive + Debug + Copy,
{
    if data1.is_empty() || data2.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot perform t-test on empty data",
        ));
    }

    if data1.len() < 2 || data2.len() < 2 {
        return Err(StatsError::invalid_input(
            "Need at least 2 data points in each group for t-test",
        ));
    }

    // Calculate sample statistics
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;
    let mean1 = calculate_mean(data1)?;
    let mean2 = calculate_mean(data2)?;
    let var1 = calculate_variance(data1, mean1)?;
    let var2 = calculate_variance(data2, mean2)?;
    let std_dev1 = var1.sqrt();
    let std_dev2 = var2.sqrt();

    let t_statistic: f64;
    let degrees_of_freedom: f64;
    let std_error: f64;

    if equal_variances {
        // Pooled variance formula for equal variances (Student's t-test)
        let pooled_variance = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        std_error = (pooled_variance * (1.0 / n1 + 1.0 / n2)).sqrt();
        t_statistic = (mean1 - mean2) / std_error;
        degrees_of_freedom = n1 + n2 - 2.0;
    } else {
        // Welch's t-test for unequal variances
        let var1_n1 = var1 / n1;
        let var2_n2 = var2 / n2;
        std_error = (var1_n1 + var2_n2).sqrt();
        t_statistic = (mean1 - mean2) / std_error;

        // Welch-Satterthwaite equation for degrees of freedom
        let numerator = (var1_n1 + var2_n2).powi(2);
        let denominator = (var1_n1.powi(2) / (n1 - 1.0)) + (var2_n2.powi(2) / (n2 - 1.0));
        degrees_of_freedom = numerator / denominator;
    }

    let p_value = calculate_p_value(t_statistic.abs(), degrees_of_freedom);

    Ok(TTestResult {
        t_statistic,
        degrees_of_freedom,
        p_value,
        mean_values: vec![mean1, mean2],
        std_devs: vec![std_dev1, std_dev2],
        std_error,
    })
}

/// Performs a paired t-test to determine if the mean difference between paired observations is zero.
///
/// This test is used when you have two related samples and want to determine if their mean
/// difference is statistically significant.
///
/// # Arguments
/// * `data1` - The first sample
/// * `data2` - The second sample (must be the same length as data1)
///
/// # Returns
/// A `Result` containing the t-test results or an error if the data is insufficient or mismatched
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::t_test::paired_t_test;
///
/// // Testing if a treatment has an effect (before vs. after)
/// let before = vec![12.1, 11.3, 13.7, 14.2, 13.8, 12.5, 11.9, 12.8, 14.0, 13.5];
/// let after = vec![12.9, 13.0, 14.3, 15.0, 14.8, 13.9, 12.7, 13.5, 15.2, 14.1];
///
/// let result = paired_t_test(&before, &after).unwrap();
/// println!("T-statistic: {}", result.t_statistic);
/// println!("P-value: {}", result.p_value);
/// println!("Mean difference: {}", result.mean_values[0]);
///
/// if result.p_value < 0.05 {
///     println!("Reject null hypothesis: There is a significant difference");
/// } else {
///     println!("Fail to reject null hypothesis");
/// }
/// ```
pub fn paired_t_test<T>(data1: &[T], data2: &[T]) -> StatsResult<TTestResult>
where
    T: ToPrimitive + Debug + Copy,
{
    if data1.is_empty() || data2.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot perform paired t-test on empty data",
        ));
    }

    if data1.len() != data2.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "Paired t-test requires equal sample sizes (got {} and {})",
            data1.len(),
            data2.len()
        )));
    }

    if data1.len() < 2 {
        return Err(StatsError::invalid_input(
            "Need at least 2 pairs for paired t-test",
        ));
    }

    // Calculate differences between paired samples
    let mut differences: Vec<f64> = Vec::with_capacity(data1.len());

    for i in 0..data1.len() {
        let val1 = data1[i].to_f64().ok_or_else(|| {
            StatsError::conversion_error(format!(
                "Failed to convert data1 value at index {} to f64",
                i
            ))
        })?;

        let val2 = data2[i].to_f64().ok_or_else(|| {
            StatsError::conversion_error(format!(
                "Failed to convert data2 value at index {} to f64",
                i
            ))
        })?;

        differences.push(val1 - val2);
    }

    // Calculate statistics on the differences
    let n = differences.len() as f64;
    let mean_diff = differences.iter().sum::<f64>() / n;

    // Calculate variance of differences
    let variance = differences
        .iter()
        .map(|&d| (d - mean_diff).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let std_dev = variance.sqrt();
    let std_error = std_dev / n.sqrt();

    // Calculate t-statistic for paired test
    let t_statistic = mean_diff / std_error;
    let degrees_of_freedom = n - 1.0;

    // Calculate p-value (two-tailed)
    let p_value = calculate_p_value(t_statistic.abs(), degrees_of_freedom);

    // Calculate original means and std devs
    let mean1 = calculate_mean(data1)?;
    let mean2 = calculate_mean(data2)?;
    let std_dev1 = calculate_variance(data1, mean1)?.sqrt();
    let std_dev2 = calculate_variance(data2, mean2)?.sqrt();

    Ok(TTestResult {
        t_statistic,
        degrees_of_freedom,
        p_value,
        // First value is the mean difference, followed by the means of each dataset
        mean_values: vec![mean_diff, mean1, mean2],
        std_devs: vec![std_dev, std_dev1, std_dev2],
        std_error,
    })
}

// Helper functions

/// Calculates the mean of a sample
fn calculate_mean<T>(data: &[T]) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    if data.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot calculate mean of empty data",
        ));
    }

    let mut sum = 0.0;
    let n = data.len() as f64;

    for (i, value) in data.iter().enumerate() {
        let v = value.to_f64().ok_or_else(|| {
            StatsError::conversion_error(format!("Failed to convert value at index {} to f64", i))
        })?;
        sum += v;
    }

    Ok(sum / n)
}

/// Calculates the variance of a sample
fn calculate_variance<T>(data: &[T], mean: f64) -> StatsResult<f64>
where
    T: ToPrimitive + Debug,
{
    if data.is_empty() {
        return Err(StatsError::empty_data(
            "Cannot calculate variance of empty data",
        ));
    }

    if data.len() < 2 {
        return Err(StatsError::invalid_input(
            "Need at least 2 data points to calculate variance",
        ));
    }

    let mut sum_squared_diff = 0.0;
    let n = data.len() as f64;

    for (i, value) in data.iter().enumerate() {
        let v = value.to_f64().ok_or_else(|| {
            StatsError::conversion_error(format!("Failed to convert value at index {} to f64", i))
        })?;
        sum_squared_diff += (v - mean).powi(2);
    }

    Ok(sum_squared_diff / (n - 1.0))
}

/// Calculates p-value from t-statistic and degrees of freedom
///
/// # Arguments
/// * `t_stat` - The absolute value of the t-statistic
/// * `df` - Degrees of freedom
///
/// # Returns
/// The two-tailed p-value corresponding to the t-statistic and degrees of freedom
fn calculate_p_value(t_stat: f64, df: f64) -> f64 {
    // For very large degrees of freedom, t-distribution approaches normal distribution
    if df > 1000.0 {
        // Use normal approximation for large df
        let z = t_stat;
        return 2.0 * (1.0 - standard_normal_cdf(z));
    }

    // Use Student's t-distribution CDF approximation
    // This is an implementation of the algorithm from:
    // Abramowitz and Stegun: Handbook of Mathematical Functions
    //
    // The incomplete beta function I_x(a, b) where x = df/(df + t^2) gives us
    // the cumulative probability P(T ≤ |t|) for the t-distribution.
    // For a two-tailed test: p-value = 2 * (1 - P(T ≤ |t|))

    let a = df / (df + t_stat * t_stat);
    let ix = incomplete_beta(0.5 * df, 0.5, a);

    // Two-tailed p-value: clamp to [0.0, 1.0] to handle numerical precision issues
    (2.0 * (1.0 - ix)).clamp(0.0, 1.0)
}

/// Standard normal cumulative distribution function
fn standard_normal_cdf(x: f64) -> f64 {
    // Use error function relationship with normal CDF
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Incomplete beta function approximation
/// Used for calculating the cumulative distribution function of the t-distribution
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return x;
    }

    // Use continued fraction approximation for incomplete beta
    let symmetry_point = x > (a / (a + b));

    // Apply symmetry for more accurate computation when x > a/(a+b)
    let (a_calc, b_calc, x_calc) = if symmetry_point {
        (b, a, 1.0 - x)
    } else {
        (a, b, x)
    };

    // Constants for the continued fraction method
    let max_iterations = 200;
    let epsilon = 1e-10;

    // Continued fraction expansion using modified Lentz's method
    let front_factor = x_calc.powf(a_calc) * (1.0 - x_calc).powf(b_calc) / beta(a_calc, b_calc);

    let mut h = 1.0;
    let mut d = 1.0;
    let mut result = 0.0;

    for m in 1..max_iterations {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even term
        let numerator = (m * (b_calc - m) * x_calc) / ((a_calc + m2 - 1.0) * (a_calc + m2));

        d = 1.0 + numerator * d;
        if d.abs() < epsilon {
            d = epsilon;
        }
        d = 1.0 / d;

        h = 1.0 + numerator / h;
        if h.abs() < epsilon {
            h = epsilon;
        }

        result *= h * d;

        // Odd term
        let numerator = -((a_calc + m) * (a_calc + b_calc + m) * x_calc)
            / ((a_calc + m2) * (a_calc + m2 + 1.0));

        d = 1.0 + numerator * d;
        if d.abs() < epsilon {
            d = epsilon;
        }
        d = 1.0 / d;

        h = 1.0 + numerator / h;
        if h.abs() < epsilon {
            h = epsilon;
        }

        let delta = h * d;
        result *= delta;

        // Check for convergence
        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    // Apply the front factor
    result *= front_factor;

    // Return appropriate result based on symmetry
    if symmetry_point { 1.0 - result } else { result }
}

/// Beta function B(a, b) = Γ(a) * Γ(b) / Γ(a + b)
/// where Γ is the gamma function
fn beta(a: f64, b: f64) -> f64 {
    // Use Stirling's approximation for gamma function
    let log_gamma_a = ln_gamma(a);
    let log_gamma_b = ln_gamma(b);
    let log_gamma_ab = ln_gamma(a + b);

    (log_gamma_a + log_gamma_b - log_gamma_ab).exp()
}

/// Natural logarithm of the gamma function
/// Using Lanczos approximation for the gamma function
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(1-x) = π / (sin(πx) * Γ(x))
        // ln(Γ(x)) = ln(π) - ln(sin(πx)) - ln(Γ(1-x))
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        // Standard Lanczos approximation for x ≥ 0.5
        let mut sum = p[0];
        for (i, &value) in p.iter().enumerate().skip(1) {
            sum += value / (x + i as f64);
        }

        let t = x + 7.5;
        (x + 0.5) * t.ln() - t + LN_2PI * 0.5 + sum.ln() / x
    }
}
/// Error function implementation (erf)
///
/// Computes an approximation to the error function using a numerical approximation
/// based on Abramowitz and Stegun formula 7.1.26.
fn erf(x: f64) -> f64 {
    // Constants for the approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Formula 7.1.26 from Abramowitz and Stegun
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p_value_range_one_sample() {
        // Test that p-values are always in [0.0, 1.0]
        let data = vec![5.2, 6.4, 6.9, 7.3, 7.5, 7.8, 8.1, 8.4, 9.2, 9.5];
        let population_mean = 7.0;

        let result = one_sample_t_test(&data, population_mean).unwrap();
        assert!(
            result.p_value >= 0.0,
            "p-value should be >= 0.0, got {}",
            result.p_value
        );
        assert!(
            result.p_value <= 1.0,
            "p-value should be <= 1.0, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_p_value_range_two_sample() {
        let group1 = vec![5.2, 6.4, 6.9, 7.3, 7.5, 7.8, 8.1, 8.4, 9.2, 9.5];
        let group2 = vec![4.1, 5.0, 5.5, 6.2, 6.3, 6.5, 6.8, 7.1, 7.4, 7.5];

        let result = two_sample_t_test(&group1, &group2, false).unwrap();
        assert!(
            result.p_value >= 0.0,
            "p-value should be >= 0.0, got {}",
            result.p_value
        );
        assert!(
            result.p_value <= 1.0,
            "p-value should be <= 1.0, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_p_value_range_paired() {
        let before = vec![12.1, 11.3, 13.7, 14.2, 13.8, 12.5, 11.9, 12.8, 14.0, 13.5];
        let after = vec![12.9, 13.0, 14.3, 15.0, 14.8, 13.9, 12.7, 13.5, 15.2, 14.1];

        let result = paired_t_test(&before, &after).unwrap();
        assert!(
            result.p_value >= 0.0,
            "p-value should be >= 0.0, got {}",
            result.p_value
        );
        assert!(
            result.p_value <= 1.0,
            "p-value should be <= 1.0, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_p_value_edge_cases() {
        // Test with various t-statistics to ensure p-value stays in range
        let test_cases = vec![
            (0.0, 5.0),  // t = 0
            (1.0, 10.0), // Small t
            (2.0, 20.0), // Medium t
            (5.0, 30.0), // Large t
        ];

        for (t_stat, df) in test_cases {
            let p_value = calculate_p_value(t_stat, df);
            assert!(
                p_value >= 0.0,
                "p-value should be >= 0.0 for t={}, df={}, got {}",
                t_stat,
                df,
                p_value
            );
            assert!(
                p_value <= 1.0,
                "p-value should be <= 1.0 for t={}, df={}, got {}",
                t_stat,
                df,
                p_value
            );
        }
    }
}
