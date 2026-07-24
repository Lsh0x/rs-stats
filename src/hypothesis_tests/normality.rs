//! # D'Agostino-Pearson K² normality test
//!
//! Omnibus test of the null hypothesis that data come from a normal
//! distribution, combining a skewness test (D'Agostino 1970) and a
//! kurtosis test (Anscombe & Glynn 1983): `K² = z₁² + z₂² ~ χ²(2)` under
//! H₀. Matches `scipy.stats.normaltest` (and `skewtest` / `kurtosistest`)
//! exactly. Run it before reaching for a t-test — and reach for
//! [`mann_whitney_u`](crate::hypothesis_tests::mann_whitney_u) when it
//! rejects.

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;

/// Result of the D'Agostino-Pearson normality test.
#[derive(Debug, Clone, Copy)]
pub struct NormalityResult {
    /// K² omnibus statistic (`z₁² + z₂²`).
    pub statistic: f64,
    /// p-value from χ²(2): small ⇒ reject normality.
    pub p_value: f64,
    /// z-score of the skewness test (sign = direction of the skew).
    pub skew_z: f64,
    /// z-score of the kurtosis test (positive = heavier tails than normal).
    pub kurtosis_z: f64,
}

/// D'Agostino-Pearson K² test for departure from normality.
///
/// Requires `n ≥ 8` (the transformations below are calibrated for that
/// regime; scipy enforces the same bound).
///
/// # Examples
/// ```
/// use rs_stats::hypothesis_tests::dagostino_k2;
///
/// // Right-skewed data with an outlier: normality is rejected.
/// let x = [2.1, 3.4, 1.8, 2.9, 3.1, 2.5, 4.8, 2.2, 3.0, 2.7,
///          1.9, 3.3, 2.4, 5.6, 2.8, 3.2, 2.0, 2.6, 3.7, 2.3,
///          8.1, 3.5, 2.95, 3.05, 2.45];
/// let res = dagostino_k2(&x).unwrap();
/// assert!(res.p_value < 1e-4);
/// ```
pub fn dagostino_k2<T>(data: &[T]) -> StatsResult<NormalityResult>
where
    T: ToPrimitive,
{
    let n = data.len();
    if n < 8 {
        return Err(StatsError::invalid_input(format!(
            "dagostino_k2: need at least 8 observations, got {n}"
        )));
    }
    let xs: Vec<f64> = data
        .iter()
        .map(|v| v.to_f64())
        .collect::<Option<_>>()
        .ok_or_else(|| StatsError::conversion_error("dagostino_k2: data not convertible to f64"))?;

    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let (mut m2, mut m3, mut m4) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &x in &xs {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    if m2 == 0.0 {
        return Err(StatsError::invalid_input(
            "dagostino_k2: data has zero variance",
        ));
    }
    let g1 = m3 / m2.powf(1.5); // sample skewness
    let b2 = m4 / (m2 * m2); // sample kurtosis (not excess)

    // ── Skewness z (D'Agostino 1970, as in scipy.stats.skewtest) ──────
    let y = g1 * ((nf + 1.0) * (nf + 3.0) / (6.0 * (nf - 2.0))).sqrt();
    let beta2 = 3.0 * (nf * nf + 27.0 * nf - 70.0) * (nf + 1.0) * (nf + 3.0)
        / ((nf - 2.0) * (nf + 5.0) * (nf + 7.0) * (nf + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let y = if y == 0.0 { 1e-100 } else { y }; // scipy's guard against ln(0)
    let ya = y / alpha;
    let skew_z = delta * (ya + (ya * ya + 1.0).sqrt()).ln();

    // ── Kurtosis z (Anscombe & Glynn 1983, as in scipy.stats.kurtosistest) ──
    let e = 3.0 * (nf - 1.0) / (nf + 1.0);
    let var_b2 =
        24.0 * nf * (nf - 2.0) * (nf - 3.0) / ((nf + 1.0) * (nf + 1.0) * (nf + 3.0) * (nf + 5.0));
    let x_std = (b2 - e) / var_b2.sqrt();
    let sqrt_beta1 = 6.0 * (nf * nf - 5.0 * nf + 2.0) / ((nf + 7.0) * (nf + 9.0))
        * (6.0 * (nf + 3.0) * (nf + 5.0) / (nf * (nf - 2.0) * (nf - 3.0))).sqrt();
    let a = 6.0
        + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + (1.0 + 4.0 / (sqrt_beta1 * sqrt_beta1)).sqrt());
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let denom = 1.0 + x_std * (2.0 / (a - 4.0)).sqrt();
    // scipy handles denom < 0 via a signed cbrt.
    let term2 = ((1.0 - 2.0 / a) / denom.abs()).cbrt() * denom.signum();
    let kurtosis_z = (term1 - term2) / (2.0 / (9.0 * a)).sqrt();

    // ── Omnibus: K² ~ χ²(2) ⇒ p = exp(−K²/2) exactly ──────────────────
    let k2 = skew_z * skew_z + kurtosis_z * kurtosis_z;
    let p_value = (-0.5 * k2).exp().clamp(0.0, 1.0);

    Ok(NormalityResult {
        statistic: k2,
        p_value,
        skew_z,
        kurtosis_z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.normaltest / skewtest / kurtosistest.
    const X: [f64; 25] = [
        2.1, 3.4, 1.8, 2.9, 3.1, 2.5, 4.8, 2.2, 3.0, 2.7, 1.9, 3.3, 2.4, 5.6, 2.8, 3.2, 2.0, 2.6,
        3.7, 2.3, 8.1, 3.5, 2.95, 3.05, 2.45,
    ];

    #[test]
    fn test_matches_scipy_skewed() {
        let r = dagostino_k2(&X).unwrap();
        assert!(
            (r.statistic - 29.23244472395368).abs() < 1e-10,
            "{}",
            r.statistic
        );
        assert!((r.p_value - 4.4900924376021633e-7).abs() < 1e-16);
        assert!((r.skew_z - 4.122491219672165).abs() < 1e-10);
        assert!((r.kurtosis_z - 3.4982153832603826).abs() < 1e-10);
    }

    #[test]
    fn test_matches_scipy_linear() {
        // Evenly-spaced data: platykurtic but symmetric — not rejected at 5%.
        let y: Vec<f64> = (0..30).map(|i| -2.0 + 4.0 * i as f64 / 29.0).collect();
        let r = dagostino_k2(&y).unwrap();
        assert!((r.statistic - 5.419188147156354).abs() < 1e-10);
        assert!((r.p_value - 0.0665638212454711).abs() < 1e-12);
    }

    #[test]
    fn test_errors() {
        assert!(dagostino_k2(&[1.0; 7]).is_err()); // n < 8
        assert!(dagostino_k2(&[3.0; 20]).is_err()); // zero variance
    }
}
