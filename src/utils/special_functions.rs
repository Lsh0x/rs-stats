//! # Special Mathematical Functions
//!
//! Provides the core special functions required by statistical distributions:
//! - Natural logarithm of the Gamma function (Lanczos approximation)
//! - Regularized incomplete gamma function P(a, x) — used by Gamma/ChiSquared CDF
//! - Regularized incomplete beta function I_x(a, b) — used by Beta/StudentT/F CDF

use crate::utils::constants::{LN_2PI, PI};

// ── Gamma / Log-Gamma ─────────────────────────────────────────────────────────

/// Natural logarithm of the Gamma function using the Lanczos approximation.
///
/// Accurate to ~15 significant figures for `x > 0`.
/// Uses the reflection formula for `0 < x < 0.5`.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients for g = 7, n = 9
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x) · Γ(1-x) = π / sin(πx)
        PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut s = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            s += ci / (z + (i as f64) + 1.0);
        }
        let t = z + G + 0.5;
        LN_2PI * 0.5 + (z + 0.5) * t.ln() - t + s.ln()
    }
}

/// Gamma function Γ(x) = exp(ln_gamma(x)).
#[inline]
pub fn gamma_fn(x: f64) -> f64 {
    ln_gamma(x).exp()
}

// ── Regularized Incomplete Gamma ───────────────────────────────────────────────

/// Regularized lower incomplete gamma function P(a, x) = γ(a,x) / Γ(a).
///
/// Returns the probability that a Gamma(a, 1) random variable is ≤ x.
///
/// # Arguments
/// * `a` - shape parameter (must be > 0)
/// * `x` - upper limit of integration (must be ≥ 0)
pub fn regularized_incomplete_gamma(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0, "a must be positive");
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_continued_fraction(a, x)
    }
}

/// Regularized *upper* incomplete gamma function Q(a, x) = 1 − P(a, x).
///
/// Computed directly (series or continued fraction on the appropriate side)
/// so tiny tail values keep full **relative** precision — `1.0 - P(a, x)`
/// would round to 0 as soon as Q < ~1e-16. This is the primitive behind
/// accurate tail probabilities (`erfc`, survival functions, p-values).
pub fn regularized_incomplete_gamma_upper(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0, "a must be positive");
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_continued_fraction(a, x)
    }
}

/// Series representation of the lower regularized incomplete gamma.
/// Converges well when x < a + 1.
fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 300;
    let eps = 1e-12;
    let log_factor = -x + a * x.ln() - ln_gamma(a);

    let mut term = 1.0 / a;
    let mut sum = term;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    (sum * log_factor.exp()).clamp(0.0, 1.0)
}

/// Continued-fraction representation of the *upper* regularized incomplete gamma Q(a, x).
/// Converges well when x >= a + 1.  Returns Q = 1 - P.
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let max_iter = 300;
    let eps = 1e-12;
    let fpmin = 1e-300_f64;

    let log_factor = -x + a * x.ln() - ln_gamma(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = (an * d + b).max(fpmin);
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    (h * log_factor.exp()).clamp(0.0, 1.0)
}

// ── Beta function ──────────────────────────────────────────────────────────────

/// Beta function B(a, b) = Γ(a) · Γ(b) / Γ(a + b).
#[inline]
pub fn beta_fn(a: f64, b: f64) -> f64 {
    (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)).exp()
}

/// ln B(a, b).
#[inline]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

// ── Regularized Incomplete Beta ────────────────────────────────────────────────

/// Regularized incomplete beta function I_x(a, b) = B(x; a, b) / B(a, b).
///
/// Returns the probability that a Beta(a, b) random variable is ≤ x.
///
/// Uses Numerical Recipes' `betacf` continued-fraction (Lentz's method) with
/// the standard NR symmetry condition `x < (a+1)/(a+b+2)`.
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // bt = x^a * (1-x)^b / B(a,b)  — shared log-space factor
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    // NR symmetry: use the CF that converges faster
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Continued-fraction core `betacf(a, b, x)` (Numerical Recipes, Press et al.).
///
/// Returns the value h of the CF such that I_x(a,b) = bt * h / a.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 300;
    let eps = 3e-12;
    let fpmin = 1e-300_f64;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    // First step: d = 1 / (1 - qab*x/qap)
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..max_iter {
        let mf = m as f64;
        let m2 = 2.0 * mf;

        // Even step: aa = m*(b-m)*x / ((qam+m2)*(a+m2))
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step: aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

// ── Noncentral Student-t CDF (Lenth 1989) ─────────────────────────────────────

/// CDF of the noncentral t distribution: `P(T ≤ t)` with `df` degrees of
/// freedom and noncentrality `nc`.
///
/// Lenth's series (Algorithm AS 243): with `x = t²/(t²+df)`,
/// `P = Φ(−nc) + ½ Σⱼ [pⱼ·I_x(j+½, df/2) + qⱼ·I_x(j+1, df/2)]` where the
/// Poisson-like weights `pⱼ`, `qⱼ` are computed by recurrence. This is the
/// engine behind exact statistical power calculations
/// (see [`crate::hypothesis_tests::power_t_test`]).
pub fn noncentral_t_cdf(t: f64, df: f64, nc: f64) -> f64 {
    debug_assert!(df > 0.0, "df must be positive");
    // Symmetry reduces to t ≥ 0: P(T ≤ t; δ) = 1 − P(T ≤ −t; −δ).
    if t < 0.0 {
        return 1.0 - noncentral_t_cdf(-t, df, -nc);
    }
    let phi_neg_nc = 0.5 * crate::prob::erf::erfc_cody(nc / std::f64::consts::SQRT_2);
    if t == 0.0 {
        return phi_neg_nc;
    }
    let half_nc2 = 0.5 * nc * nc;
    if half_nc2 > 700.0 {
        // exp(−δ²/2) underflows the series weights. Fall back to the
        // Abramowitz-Stegun 26.7.10 normal approximation, which keeps the
        // t-dependence (a flat 0/1 return here was wrong for t ≈ δ).
        let z = (t * (1.0 - 1.0 / (4.0 * df)) - nc) / (1.0 + t * t / (2.0 * df)).sqrt();
        return (0.5 * crate::prob::erf::erfc_cody(-z / std::f64::consts::SQRT_2)).clamp(0.0, 1.0);
    }

    let x = t * t / (t * t + df);
    let half_df = 0.5 * df;

    // Weights by recurrence: p₀ = e^{−δ²/2}, q₀ = δ·e^{−δ²/2}·√(2/π).
    let mut p_j = (-half_nc2).exp();
    let mut q_j = nc * (-half_nc2).exp() * (2.0 / std::f64::consts::PI).sqrt();
    let mut sum = 0.0_f64;
    let mut j = 0usize;
    loop {
        let jf = j as f64;
        let term = p_j * regularized_incomplete_beta(jf + 0.5, half_df, x)
            + q_j * regularized_incomplete_beta(jf + 1.0, half_df, x);
        sum += term;
        // The Poisson weights peak near j ≈ δ²/2; only stop once past it.
        if (jf > half_nc2 && term.abs() < 1e-14 * (sum.abs() + 1e-300)) || j > 2000 {
            break;
        }
        p_j *= half_nc2 / (jf + 1.0);
        q_j *= half_nc2 / (jf + 1.5);
        j += 1;
    }

    (phi_neg_nc + 0.5 * sum).clamp(0.0, 1.0)
}

/// Survival function of the noncentral t: `P(T > t)`.
pub fn noncentral_t_sf(t: f64, df: f64, nc: f64) -> f64 {
    // Recompute on the mirrored side rather than 1 − cdf so small upper
    // tails keep relative precision.
    noncentral_t_cdf(-t, df, -nc)
}

// ── Inverse CDF via bracketed Newton ──────────────────────────────────────────

/// Find x with `cdf_fn(x) ≈ p` using a short bisection to localise the
/// root, then bracket-safeguarded Newton steps (`x ← x − (F(x)−p)/f(x)`).
///
/// Quadratic convergence cuts the CDF evaluations from ~60 (pure
/// bisection at relative 1e-12) to ~15. Falls back to a bisection step
/// whenever Newton leaves the bracket or the density vanishes, so it is
/// as robust as [`bisect_inverse_cdf`] (including the same dynamic
/// bracket expansion).
pub fn newton_inverse_cdf(
    cdf_fn: impl Fn(f64) -> f64,
    pdf_fn: impl Fn(f64) -> f64,
    p: f64,
    mut lo: f64,
    mut hi: f64,
) -> f64 {
    const EPS: f64 = 1e-12;

    // Same dynamic bracket expansion as bisect_inverse_cdf.
    let mut width = (hi - lo).abs().max(1.0);
    for _ in 0..200 {
        if cdf_fn(hi) >= p {
            break;
        }
        lo = hi;
        hi += width;
        width *= 2.0;
    }
    let mut width = (hi - lo).abs().max(1.0);
    for _ in 0..200 {
        if cdf_fn(lo) <= p {
            break;
        }
        hi = lo;
        lo -= width;
        width *= 2.0;
    }

    // A few bisection steps to give Newton a well-conditioned start.
    for _ in 0..6 {
        let mid = 0.5 * (lo + hi);
        if cdf_fn(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear bisection cannot resolve roots far below the bracket width:
    // with lo pinned at 0, every root under ~1e-12 used to "converge" to
    // the bracket edge (e.g. Gamma(0.01, 1) medians ~1e-31). Locate tiny
    // roots' order of magnitude in log scale first.
    if lo == 0.0 && hi > 0.0 {
        let tiny = f64::MIN_POSITIVE;
        if cdf_fn(tiny) >= p {
            return tiny; // the root sits below the smallest positive double
        }
        let mut glo = tiny;
        let mut ghi = hi;
        for _ in 0..80 {
            let mid = (0.5 * (glo.ln() + ghi.ln())).exp();
            if cdf_fn(mid) < p {
                glo = mid;
            } else {
                ghi = mid;
            }
            if ghi <= glo * (1.0 + 1e-12) {
                return 0.5 * (glo + ghi);
            }
        }
        lo = glo;
        hi = ghi;
    }

    let mut x = 0.5 * (lo + hi);
    for _ in 0..40 {
        let f = cdf_fn(x) - p;
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let d = pdf_fn(x);
        let mut x_new = x - f / d;
        if !x_new.is_finite() || x_new <= lo || x_new >= hi {
            // Newton left the bracket (flat density, overshoot): bisect.
            x_new = 0.5 * (lo + hi);
        }
        // Relative-only convergence — an absolute floor here is exactly
        // what silently truncated sub-1e-12 roots.
        if x_new == x || (x_new - x).abs() <= EPS * x_new.abs() {
            return x_new;
        }
        x = x_new;
        if (hi - lo).abs() <= EPS * hi.abs().max(lo.abs()) {
            return x;
        }
    }
    x
}

// ── General inverse CDF via bisection ─────────────────────────────────────────

/// Find x such that `cdf_fn(x) ≈ p` via bisection starting from `[lo, hi]`.
/// `cdf_fn` must be monotone non-decreasing.
///
/// `[lo, hi]` is a *seed* bracket, not a hard bound: if the quantile lies
/// outside it, the bracket is expanded geometrically until it contains `p`
/// (heavy-tailed distributions — t with ν ≤ 2, F with small d2 — put
/// extreme quantiles orders of magnitude beyond any moment-based guess).
/// Bounded supports never expand: their endpoint CDF already reaches 0/1.
pub fn bisect_inverse_cdf(cdf_fn: impl Fn(f64) -> f64, p: f64, mut lo: f64, mut hi: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-12;

    let mut width = (hi - lo).abs().max(1.0);
    for _ in 0..MAX_ITER {
        if cdf_fn(hi) >= p {
            break;
        }
        lo = hi;
        hi += width;
        width *= 2.0;
    }
    let mut width = (hi - lo).abs().max(1.0);
    for _ in 0..MAX_ITER {
        if cdf_fn(lo) <= p {
            break;
        }
        hi = lo;
        lo -= width;
        width *= 2.0;
    }

    for _ in 0..MAX_ITER {
        let mid = 0.5 * (lo + hi);
        // Relative tolerance: far-tail quantiles can be huge and an absolute
        // 1e-12 is then below one ulp (the loop would spin to MAX_ITER).
        if (hi - lo).abs() < EPS * mid.abs().max(1.0) {
            return mid;
        }
        if cdf_fn(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln_gamma_known_values() {
        // Γ(1) = 1  → ln_gamma(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // Γ(2) = 1  → ln_gamma(2) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // Γ(0.5) = sqrt(π) → ln_gamma(0.5) = 0.5*ln(π)
        let expected = 0.5 * PI.ln();
        assert!((ln_gamma(0.5) - expected).abs() < 1e-10);
        // Γ(5) = 4! = 24
        assert!((ln_gamma(5.0) - 24_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_regularized_incomplete_gamma() {
        // P(1, 0) = 0, P(1, inf) → 1
        assert_eq!(regularized_incomplete_gamma(1.0, 0.0), 0.0);
        // P(1, x) = 1 - e^(-x) for Gamma(1) = Exponential(1)
        let x = 2.0_f64;
        let expected = 1.0 - (-x).exp();
        assert!((regularized_incomplete_gamma(1.0, x) - expected).abs() < 1e-8);
    }

    #[test]
    fn test_regularized_incomplete_beta() {
        // I_x(1,1) = x  (uniform on [0,1])
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                (regularized_incomplete_beta(1.0, 1.0, x) - x).abs() < 1e-9,
                "I_{}(1,1) should be {}",
                x,
                x
            );
        }
        // Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        let (a, b, x) = (2.0, 3.0, 0.4);
        let lhs = regularized_incomplete_beta(a, b, x);
        let rhs = 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
        assert!((lhs - rhs).abs() < 1e-10);
    }
}
