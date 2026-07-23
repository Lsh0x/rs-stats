//! # Error Function (erf)
//!
//! This module implements the error function, a special function that occurs in probability,
//! statistics, and partial differential equations.
//!
//! ## Mathematical Definition
//! The error function is defined as:
//!
//! erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
//!
//! ## Key Properties
//! - erf(-x) = -erf(x) (odd function)
//! - erf(0) = 0
//! - erf(∞) = 1
//! - erf(-∞) = -1
//!
//! ## Implementation Details
//! Computed with W. J. Cody's rational Chebyshev approximations (the
//! CALERF algorithm, 1969/1990 — the same scheme used by glibc and
//! Fortran SPECFUN): three regions, each a small fixed-degree rational,
//! accurate to ~1 ulp across the real line. Replaces the previous
//! delegation to the *iterative* regularized incomplete gamma (up to 300
//! series/continued-fraction steps per call) — same accuracy, an order
//! of magnitude faster on the `Normal::cdf` hot path.

use crate::error::{StatsError, StatsResult};
use num_traits::ToPrimitive;

// ── Cody / CALERF rational approximations ─────────────────────────────────────

/// Region boundary: below this |x| use the erf rational, above it the
/// erfc rationals.
const THRESH: f64 = 0.46875;
/// erfc underflows to 0 beyond this point (exp(−x²) < DBL_MIN).
const XBIG: f64 = 26.543;
/// 1/√π.
const SQRPI: f64 = 5.641_895_835_477_562_9e-1;

/// erf numerator/denominator, |x| ≤ 0.46875.
const A: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_56e2,
    3.774_852_376_853_020_2e2,
    3.209_377_589_138_469_47e3,
    1.857_777_061_846_031_53e-1,
];
const B: [f64; 4] = [
    2.360_129_095_234_412_09e1,
    2.440_246_379_344_441_73e2,
    1.282_616_526_077_372_28e3,
    2.844_236_833_439_170_62e3,
];
/// erfc·exp(x²) numerator/denominator, 0.46875 < x ≤ 4.
const C: [f64; 9] = [
    5.641_884_969_886_700_89e-1,
    8.883_149_794_388_375_94e0,
    6.611_919_063_714_162_95e1,
    2.986_351_381_974_001_31e2,
    8.819_522_212_417_690_9e2,
    1.712_047_612_634_070_58e3,
    2.051_078_377_826_071_47e3,
    1.230_339_354_797_997_25e3,
    2.153_115_354_744_038_46e-8,
];
const D: [f64; 8] = [
    1.574_492_611_070_983_47e1,
    1.176_939_508_913_124_99e2,
    5.371_811_018_620_098_58e2,
    1.621_389_574_566_690_19e3,
    3.290_799_235_733_459_63e3,
    4.362_619_090_143_247_16e3,
    3.439_367_674_143_721_64e3,
    1.230_339_354_803_749_42e3,
];
/// erfc·x·exp(x²) − 1/√π correction, x > 4.
const P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
const Q: [f64; 5] = [
    2.568_520_192_289_822_42e0,
    1.872_952_849_923_460_47e0,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];

/// Region 1: erf(x) for |x| ≤ 0.46875 (rational in x²).
fn erf_small(x: f64) -> f64 {
    let z = x * x;
    let mut xnum = A[4] * z;
    let mut xden = z;
    for i in 0..3 {
        xnum = (xnum + A[i]) * z;
        xden = (xden + B[i]) * z;
    }
    x * (xnum + A[3]) / (xden + B[3])
}

/// Regions 2–3: erfc(y) for y > 0.46875.
fn erfc_large(y: f64) -> f64 {
    let r = if y <= 4.0 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        (xnum + C[7]) / (xden + D[7])
    } else {
        if y >= XBIG {
            return 0.0;
        }
        let z = 1.0 / (y * y);
        let mut xnum = P[5] * z;
        let mut xden = z;
        for i in 0..4 {
            xnum = (xnum + P[i]) * z;
            xden = (xden + Q[i]) * z;
        }
        (SQRPI - z * (xnum + P[4]) / (xden + Q[4])) / y
    };
    // exp(−y²) via the split y² = ysq² + del with ysq on a 1/16 grid:
    // ysq·ysq is exact in f64, so the product of the two exps keeps
    // full relative precision (a direct exp(−y²) loses ~y²·ulp).
    let ysq = (y * 16.0).trunc() / 16.0;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp() * r
}

/// erf via Cody's rationals; ~1 ulp over the real line.
pub(crate) fn erf_cody(x: f64) -> f64 {
    if x.abs() <= THRESH {
        return erf_small(x);
    }
    let e = erfc_large(x.abs());
    if x >= 0.0 { 1.0 - e } else { e - 1.0 }
}

/// erfc via Cody's rationals — full *relative* precision in the upper
/// tail (down to ~1e-308), where `1 − erf` collapses.
pub(crate) fn erfc_cody(x: f64) -> f64 {
    if x.abs() <= THRESH {
        return 1.0 - erf_small(x);
    }
    let e = erfc_large(x.abs());
    if x >= 0.0 { e } else { 2.0 - e }
}

/// Calculate the error function (erf) of a value
///
/// The error function is related to the normal distribution and is used
/// in probability calculations.
///
/// # Arguments
/// * `x` - The value at which to evaluate the error function
///
/// # Returns
/// The value of the error function at x
///
/// # Examples
/// ```
/// use rs_stats::prob::erf;
///
/// // Calculate erf(1.0)
/// let x: f64 = 1.0;
/// let result = erf(x).unwrap();
/// assert!((result - 0.842_700_792_949_714_9).abs() < 1e-12);
///
/// // Verify symmetry property
/// assert!((erf(x).unwrap() + erf(-x).unwrap()).abs() < 1e-12);
/// ```
#[inline]
pub fn erf<T>(x: T) -> StatsResult<f64>
where
    T: ToPrimitive,
{
    let x = x.to_f64().ok_or_else(|| StatsError::ConversionError {
        message: "prob::erf: Failed to convert x to f64".to_string(),
    })?;
    if x == 0.0 {
        return Ok(0.0);
    }
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x.is_infinite() {
        return Ok(if x > 0.0 { 1.0 } else { -1.0 });
    }
    Ok(erf_cody(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_special_cases() {
        assert!((erf(f64::INFINITY).unwrap() - 1.0).abs() < 1e-10);
        assert!((erf(f64::NEG_INFINITY).unwrap() + 1.0).abs() < 1e-10);
        assert!(erf(f64::NAN).unwrap().is_nan());
    }

    #[test]
    fn test_erf_against_known_values() {
        let test_cases = vec![
            (-3.0, -0.999977909503),
            (-2.0, -0.995322265019),
            (-1.0, -0.842700792950),
            (0.0, 0.0),
            (0.5, 0.520499877813),
            (1.0, 0.842700792950),
            (2.0, 0.995322265019),
            (3.0, 0.999977909503),
        ];

        for (x, expected) in test_cases {
            let actual = erf(x).unwrap();
            assert!(
                (actual - expected).abs() < 1e-6,
                "For x = {}, expected {}, but got {}",
                x,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_erf_symmetry() {
        let x = 0.7;
        let actual = erf(x).unwrap() + erf(-x).unwrap();
        assert!(
            actual.abs() < 1e-10,
            "erf(x) + erf(-x) should be 0.0, but got {}",
            actual
        );
    }

    #[test]
    fn test_erf_limits() {
        // Test erf approaching its limits
        assert!((erf(10.0).unwrap() - 1.0).abs() < 1e-15); // erf(x) -> 1 as x -> +inf
        assert!((erf(-10.0).unwrap() + 1.0).abs() < 1e-15); // erf(x) -> -1 as x -> -inf
    }

    #[test]
    fn test_erf_large_negative() {
        let x = -8.0;
        let actual = erf(x).unwrap();
        assert!(
            (actual + 1.0).abs() < 1e-10,
            "For large negative x, erf(x) should be close to -1.0, but got {}",
            actual
        );
    }
}
