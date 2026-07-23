//! # Correlation coefficients
//!
//! Pearson (linear) and Spearman (rank) correlation, each with a
//! two-sided significance test based on the Student-t approximation
//! `t = r·√((n−2)/(1−r²))` — the same conventions as
//! `scipy.stats.pearsonr` / `spearmanr`.

use crate::error::{StatsError, StatsResult};
use crate::utils::numeric::average_ranks;
use crate::utils::special_functions::regularized_incomplete_beta;
use num_traits::ToPrimitive;

/// Result of a correlation significance test.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationTest {
    /// Correlation coefficient in [−1, 1].
    pub r: f64,
    /// Student-t statistic `r·√((n−2)/(1−r²))`.
    pub t_statistic: f64,
    /// Two-sided p-value (H₀: no correlation).
    pub p_value: f64,
    /// Sample size.
    pub n: usize,
}

fn to_f64_pairs<X, Y>(x: &[X], y: &[Y], caller: &str) -> StatsResult<(Vec<f64>, Vec<f64>)>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "{caller}: x and y must have the same length (got {} and {})",
            x.len(),
            y.len()
        )));
    }
    if x.len() < 2 {
        return Err(StatsError::invalid_input(format!(
            "{caller}: need at least 2 paired observations, got {}",
            x.len()
        )));
    }
    let xf = x
        .iter()
        .map(|v| v.to_f64())
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| StatsError::conversion_error(format!("{caller}: x not convertible to f64")))?;
    let yf = y
        .iter()
        .map(|v| v.to_f64())
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| StatsError::conversion_error(format!("{caller}: y not convertible to f64")))?;
    Ok((xf, yf))
}

fn pearson_f64(x: &[f64], y: &[f64], caller: &str) -> StatsResult<f64> {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return Err(StatsError::invalid_input(format!(
            "{caller}: correlation is undefined when either variable is constant"
        )));
    }
    Ok((sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0))
}

/// Two-sided p-value of `t` under Student-t with `df` degrees of freedom.
fn t_two_sided_p(t: f64, df: f64) -> f64 {
    if t.is_infinite() {
        return 0.0;
    }
    regularized_incomplete_beta(df / 2.0, 0.5, df / (t * t + df)).clamp(0.0, 1.0)
}

/// Pearson product-moment correlation coefficient.
///
/// # Errors
/// Returns an error on length mismatch, fewer than 2 pairs, or when either
/// variable is constant (zero variance).
///
/// # Examples
/// ```
/// use rs_stats::prob::correlation::pearson;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.1, 5.9, 8.2, 9.8];
/// let r = pearson(&x, &y).unwrap();
/// assert!(r > 0.99);
/// ```
pub fn pearson<X, Y>(x: &[X], y: &[Y]) -> StatsResult<f64>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    let (xf, yf) = to_f64_pairs(x, y, "pearson")?;
    pearson_f64(&xf, &yf, "pearson")
}

/// Pearson correlation with a two-sided significance test.
///
/// `t = r·√((n−2)/(1−r²))` with `n − 2` degrees of freedom (matches
/// `scipy.stats.pearsonr`). Requires `n ≥ 3`.
pub fn pearson_test<X, Y>(x: &[X], y: &[Y]) -> StatsResult<CorrelationTest>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    let (xf, yf) = to_f64_pairs(x, y, "pearson_test")?;
    if xf.len() < 3 {
        return Err(StatsError::invalid_input(
            "pearson_test: need at least 3 paired observations for a significance test",
        ));
    }
    let r = pearson_f64(&xf, &yf, "pearson_test")?;
    let n = xf.len();
    let df = (n - 2) as f64;
    let t = if r.abs() >= 1.0 {
        f64::INFINITY * r.signum()
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    };
    Ok(CorrelationTest {
        r,
        t_statistic: t,
        p_value: t_two_sided_p(t, df),
        n,
    })
}

/// Spearman rank correlation coefficient `ρ`.
///
/// Ranks both variables (average ranks on ties) and computes the Pearson
/// correlation of the ranks — the standard tie-robust definition used by
/// `scipy.stats.spearmanr`.
pub fn spearman<X, Y>(x: &[X], y: &[Y]) -> StatsResult<f64>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    let (xf, yf) = to_f64_pairs(x, y, "spearman")?;
    let rx = average_ranks(&xf);
    let ry = average_ranks(&yf);
    pearson_f64(&rx, &ry, "spearman")
}

/// Spearman correlation with a two-sided significance test (Student-t
/// approximation on ρ, as in `scipy.stats.spearmanr`). Requires `n ≥ 3`.
pub fn spearman_test<X, Y>(x: &[X], y: &[Y]) -> StatsResult<CorrelationTest>
where
    X: ToPrimitive,
    Y: ToPrimitive,
{
    let (xf, yf) = to_f64_pairs(x, y, "spearman_test")?;
    if xf.len() < 3 {
        return Err(StatsError::invalid_input(
            "spearman_test: need at least 3 paired observations for a significance test",
        ));
    }
    let rx = average_ranks(&xf);
    let ry = average_ranks(&yf);
    let rho = pearson_f64(&rx, &ry, "spearman_test")?;
    let n = xf.len();
    let df = (n - 2) as f64;
    let t = if rho.abs() >= 1.0 {
        f64::INFINITY * rho.signum()
    } else {
        rho * (df / (1.0 - rho * rho)).sqrt()
    };
    Ok(CorrelationTest {
        r: rho,
        t_statistic: t,
        p_value: t_two_sided_p(t, df),
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&x, &y).unwrap() - 1.0).abs() < 1e-12);
        let y_neg = [8.0, 6.0, 4.0, 2.0];
        assert!((pearson(&x, &y_neg).unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pearson_known_value() {
        // Cross-checked with scipy.stats.pearsonr.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = [2.1, 4.0, 5.9, 8.1, 10.0, 12.2, 13.8];
        let res = pearson_test(&x, &y).unwrap();
        assert!((res.r - 0.999534510823262).abs() < 1e-12);
        // scipy.stats.pearsonr p-value: 8.9767576e-09
        assert!((res.p_value - 8.976757650992674e-9).abs() < 1e-12);
    }

    #[test]
    fn test_spearman_monotonic_nonlinear() {
        // Monotonic but nonlinear: Spearman = 1, Pearson < 1.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 8.0, 27.0, 64.0, 125.0];
        assert!((spearman(&x, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!(pearson(&x, &y).unwrap() < 1.0);
    }

    #[test]
    fn test_spearman_ties() {
        // Cross-checked with scipy.stats.spearmanr.
        let x = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 3.0, 2.0, 3.0, 5.0, 6.0];
        let rho = spearman(&x, &y).unwrap();
        assert!((rho - 0.9558823529411764).abs() < 1e-12, "rho = {rho}");
    }

    #[test]
    fn test_constant_input_errors() {
        let x = [1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0];
        assert!(pearson(&x, &y).is_err());
        assert!(pearson(&y, &x).is_err());
    }

    #[test]
    fn test_length_mismatch_and_short() {
        assert!(pearson(&[1.0, 2.0], &[1.0]).is_err());
        assert!(pearson(&[1.0], &[1.0]).is_err());
        assert!(pearson_test(&[1.0, 2.0], &[1.0, 2.0]).is_err()); // n < 3
    }
}
