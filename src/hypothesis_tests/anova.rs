//! # One-Way Analysis of Variance (ANOVA)
//!
//! One-way ANOVA is a statistical technique used to compare means of three or more independent groups.
//! It tests the null hypothesis that all group means are equal against the alternative that at least
//! one group mean is different from the others.
//!
//! ## Mathematical Formulation
//!
//! One-way ANOVA partitions the total variance in the data into:
//! - Between-group variance: Variation due to differences between group means
//! - Within-group variance: Variation due to differences within each group
//!
//! The F-statistic is calculated as the ratio of between-group variance to within-group variance:
//! F = (Between-group variance) / (Within-group variance)
//!
//! ## Assumptions
//! - Data in each group is normally distributed
//! - Groups have approximately equal variance (homoscedasticity)
//! - Observations are independent
//!
//! ## Interpretation
//! - A large F-statistic suggests the between-group variance is large compared to within-group variance,
//!   indicating significant differences between group means.
//! - The p-value indicates the probability of observing the obtained F-statistic (or more extreme)
//!   under the null hypothesis that all group means are equal.

use num_traits::ToPrimitive;
use std::fmt::Debug;

/// Result of a one-way ANOVA test
///
/// Contains the F-statistic, degrees of freedom, and p-value.
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaResult {
    /// F-statistic (ratio of between-group variance to within-group variance)
    pub f_statistic: f64,
    /// Degrees of freedom for the between-group variance (numerator)
    pub df_between: usize,
    /// Degrees of freedom for the within-group variance (denominator)
    pub df_within: usize,
    /// p-value (probability of observing the F-statistic or more extreme under the null hypothesis)
    pub p_value: f64,
    /// Sum of squares between groups
    pub ss_between: f64,
    /// Sum of squares within groups
    pub ss_within: f64,
    /// Mean square between groups
    pub ms_between: f64,
    /// Mean square within groups
    pub ms_within: f64,
}

/// Performs a one-way Analysis of Variance (ANOVA) test on multiple groups of data
///
/// Tests the null hypothesis that all group means are equal against the alternative
/// that at least one group mean is different from the others.
///
/// # Arguments
/// * `groups_data` - A slice of slices, where each inner slice contains the data for one group
///
/// # Returns
/// An `AnovaResult` struct containing the F-statistic, degrees of freedom, and p-value,
/// or `None` if any group has fewer than 2 observations or if there are fewer than 2 groups.
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::anova::one_way_anova;
///
/// // Example data for 3 groups
/// let group1 = vec![5, 7, 9, 8, 6];
/// let group2 = vec![2, 4, 3, 5, 4];
/// let group3 = vec![8, 9, 10, 7, 8];
///
/// let groups = vec![&group1[..], &group2[..], &group3[..]];
/// let result = one_way_anova(&groups).unwrap();
///
/// println!("F-statistic: {}", result.f_statistic);
/// println!("p-value: {}", result.p_value);
/// ```
pub fn one_way_anova<T>(groups_data: &[&[T]]) -> Option<AnovaResult>
where
    T: ToPrimitive + Copy + Debug,
{
    // Check if we have at least 2 groups
    if groups_data.len() < 2 {
        return None;
    }

    // Convert all data to f64 and check that each group has at least 2 observations
    let groups: Vec<Vec<f64>> = groups_data
        .iter()
        .map(|group| {
            group
                .iter()
                .filter_map(|&x| x.to_f64())
                .collect::<Vec<f64>>()
        })
        .collect();

    if groups.iter().any(|group| group.len() < 2) {
        return None;
    }

    // Calculate total number of observations
    let n_total: usize = groups.iter().map(|group| group.len()).sum();

    // Calculate the grand mean (mean of all observations)
    let all_values: Vec<f64> = groups.iter().flat_map(|group| group.iter().copied()).collect();
    let grand_mean = all_values.iter().sum::<f64>() / (n_total as f64);

    // Calculate group means
    let group_means: Vec<f64> = groups
        .iter()
        .map(|group| group.iter().sum::<f64>() / (group.len() as f64))
        .collect();

    // Calculate sum of squares between groups (SSB)
    let ss_between: f64 = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &group_mean)| {
            (group_mean - grand_mean).powi(2) * (group.len() as f64)
        })
        .sum();

    // Calculate sum of squares within groups (SSW)
    let ss_within: f64 = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &group_mean)| {
            group
                .iter()
                .map(|&value| (value - group_mean).powi(2))
                .sum::<f64>()
        })
        .sum();

    // Calculate degrees of freedom
    let df_between = groups.len() - 1;
    let df_within = n_total - groups.len();

    // Calculate mean squares
    let ms_between = ss_between / (df_between as f64);
    let ms_within = ss_within / (df_within as f64);

    // Calculate F-statistic
    let f_statistic = ms_between / ms_within;

    // Calculate p-value using the F-distribution
    let p_value = 1.0 - f_distribution_cdf(f_statistic, df_between as u32, df_within as u32);

    Some(AnovaResult {
        f_statistic,
        df_between,
        df_within,
        p_value,
        ss_between,
        ss_within,
        ms_between,
        ms_within,
    })
}

// Helper function to calculate the CDF of the F-distribution
// Uses a more accurate implementation of the regularized incomplete beta function
fn f_distribution_cdf(f: f64, df1: u32, df2: u32) -> f64 {
    // F(f; df1, df2) = I_{df2 / (df2 + df1 * f)}(df2/2, df1/2)
    // For F < 1, we can use the relationship:
    // F(f; df1, df2) = 1 - F(1/f; df2, df1)
    
    if f <= 0.0 {
        return 0.0;
    }
    
    // For F < 1, we use the complementary calculation
    if f < 1.0 {
        // Use the relationship: F(f; df1, df2) = 1 - F(1/f; df2, df1)
        return 1.0 - f_distribution_cdf(1.0 / f, df2, df1);
    }
    
    let x = df2 as f64 / (df2 as f64 + df1 as f64 * f);
    let a = df2 as f64 / 2.0;
    let b = df1 as f64 / 2.0;
    
    // Use a more accurate implementation of the regularized incomplete beta function
    regularized_incomplete_beta(x, a, b)
}

// Improved implementation of the regularized incomplete beta function
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    
    // For small values of a and b, use a continued fraction approach
    // For values where x is closer to 0, use a power series
    
    // Use a power series expansion for the incomplete beta function
    let mut term = 1.0;
    let mut sum = 0.0;
    let max_iterations = 200;
    
    // Calculate beta function normalization
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    
    for i in 0..max_iterations {
        if i > 0 {
            term *= (a + i as f64 - 1.0) * x / i as f64;
        }
        sum += term / (a + b + i as f64 - 1.0);
        if term < 1e-15 {
            break;
        }
    }
    
    (x.powf(a) * (1.0 - x).powf(b) / (-ln_beta).exp()) * sum
}

// Approximation of the natural logarithm of the gamma function
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation for ln(Gamma(x))
    if x <= 0.0 {
        return f64::INFINITY; // Not valid for non-positive numbers
    }
    
    // Coefficients for the Lanczos approximation
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ];
    
    let mut result = 0.99999999999980993;
    let z: f64 = x - 1.0;
    
    for i in 0..p.len() {
        result += p[i] / (z + (i as f64) + 1.0);
    }
    
    let t = z + p.len() as f64 - 0.5;
    (2.0 * std::f64::consts::PI).ln() / 2.0 + (t + 0.5) * t.ln() - t + result.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_way_anova_basic() {
        // Example data: three groups with clear differences
        let group1 = [5, 7, 9, 8, 6];
        let group2 = [2, 4, 3, 5, 4];
        let group3 = [8, 9, 10, 7, 8];

        let groups = [&group1[..], &group2[..], &group3[..]];
        let result = one_way_anova(&groups).unwrap();

        assert!(result.f_statistic > 1.0, "F-statistic should be greater than 1.0");
        assert!(result.p_value < 0.05, "p-value should be less than 0.05");
        assert_eq!(result.df_between, 2);
        assert_eq!(result.df_within, 12);
    }

    #[test]
    fn test_one_way_anova_equal_means() {
        // Example data: three groups with approximately equal means
        let group1 = [5, 7, 6, 5, 7];
        let group2 = [6, 5, 7, 6, 6];
        let group3 = [7, 5, 6, 7, 5];

        let groups = [&group1[..], &group2[..], &group3[..]];
        let result = one_way_anova(&groups).unwrap();

        assert!(result.f_statistic < 1.0, "F-statistic should be less than 1.0 for equal means");
        assert!(result.p_value > 0.05, "p-value should be greater than 0.05 for equal means");
    }

    #[test]
    fn test_one_way_anova_different_group_sizes() {
        // Example data: three groups with different sizes
        let group1 = [5, 7, 9, 8];
        let group2 = [2, 4, 3];
        let group3 = [8, 9, 10, 7, 8, 9];

        let groups = [&group1[..], &group2[..], &group3[..]];
        let result = one_way_anova(&groups).unwrap();

        // We're just testing that it runs without error with different group sizes
        assert!(result.df_between == 2);
        assert!(result.df_within == 10);
    }

    #[test]
    fn test_one_way_anova_float_values() {
        // Example data: three groups with float values
        let group1 = [5.2, 7.3, 9.1, 8.0, 6.5];
        let group2 = [2.1, 4.3, 3.7, 5.0, 4.2];
        let group3 = [8.1, 9.2, 10.0, 7.5, 8.3];

        let groups = [&group1[..], &group2[..], &group3[..]];
        let result = one_way_anova(&groups).unwrap();

        assert!(result.f_statistic > 1.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_one_way_anova_invalid_input() {
        // Test with empty groups
        let group1: [i32; 0] = [];
        let group2 = [1, 2, 3];
        let groups1 = [&group1[..], &group2[..]];
        assert!(one_way_anova(&groups1).is_none());

        // Test with only one group
        let groups2 = [&group2[..]];
        assert!(one_way_anova(&groups2).is_none());

        // Test with empty input
        let groups3: [&[i32]; 0] = [];
        assert!(one_way_anova(&groups3).is_none());
    }

    #[test]
    fn test_anova_result_fields() {
        // Check that all fields in AnovaResult are populated correctly
        let group1 = [5, 7, 9, 8, 6];
        let group2 = [2, 4, 3, 5, 4];
        let group3 = [8, 9, 10, 7, 8];

        let groups = [&group1[..], &group2[..], &group3[..]];
        let result = one_way_anova(&groups).unwrap();

        assert!(result.ss_between > 0.0);
        assert!(result.ss_within > 0.0);
        assert!(result.ms_between > 0.0);
        assert!(result.ms_within > 0.0);
        
        // Verify the relationship between fields
        let calculated_f = result.ms_between / result.ms_within;
        assert!((calculated_f - result.f_statistic).abs() < 1e-10);
    }
}

