extern crate num;

// Coefficients for approximation to  erf in [0, 0.84375]
const EFX: f64 = 1.283791670955e-01; // 0x3FC06EBA8214DB69
const EFX8: f64 = 1.027033336764e+00; // 0x3FF06EBA8214DB69
const PP0: f64 = 1.283791670955e-01; // 0x3FC06EBA8214DB68
const PP1: f64 = -3.250421072470e-01; // 0xBFD4CD7D691CB913
const PP2: f64 = -2.848174957559e-02; // 0xBF9D2A51DBD7194F
const PP3: f64 = -5.770270296489e-03; // 0xBF77A291236668E4
const PP4: f64 = -2.376301665665e-05; // 0xBEF8EAD6120016AC
const QQ1: f64 = 3.979172239591e-01; // 0x3FD97779CDDADC09
const QQ2: f64 = 6.502224998876e-02; // 0x3FB0A54C5536CEBA
const QQ3: f64 = 5.081306281875e-03; // 0x3F74D022C4D36B0F
const QQ4: f64 = 1.324947380043e-04; // 0x3F215DC9221C1A10
const QQ5: f64 = -3.960228278775e-06; // 0xBED09C4342A26120

// Coefficients for approximation to  erf  in [0.84375, 1.25]
const PA0: f64 = -2.362118560752e-03; // 0xBF6359B8BEF77538
const PA1: f64 = 4.148561186837e-01; // 0x3FDA8D00AD92B34D
const PA2: f64 = -3.722078760357e-01; // 0xBFD7D240FBB8C3F1
const PA3: f64 = 3.183466199011e-01; // 0x3FD45FCA805120E4
const PA4: f64 = -1.108946942823e-01; // 0xBFBC63983D3E28EC
const PA5: f64 = 3.547830432561e-02; // 0x3FA22A36599795EB
const PA6: f64 = -2.166375594868e-03; // 0xBF61BF380A96073F
const QA1: f64 = 1.064208804008e-01; // 0x3FBB3E6618EEE323
const QA2: f64 = 5.403979177021e-01; // 0x3FE14AF092EB6F33
const QA3: f64 = 7.182865441419e-02; // 0x3FB2635CD99FE9A7
const QA4: f64 = 1.261712198087e-01; // 0x3FC02660E763351F
const QA5: f64 = 1.363708391202e-02; // 0x3F8BEDC26B51DD1C
const QA6: f64 = 1.198449984679e-02; // 0x3F888B545735151D

// Coefficients for approximation to  erfc in [1.25, 1/0.35]
const RA0: f64 = -9.864944034847e-03; // 0xBF843412600D6435
const RA1: f64 = -6.938585727071e-01; // 0xBFE63416E4BA7360
const RA2: f64 = -1.055862622532e+01; // 0xC0251E0441B0E726
const RA3: f64 = -6.237533245032e+01; // 0xC04F300AE4CBA38D
const RA4: f64 = -1.623966694625e+02; // 0xC0644CB184282266
const RA5: f64 = -1.846050929067e+02; // 0xC067135CEBCCABB2
const RA6: f64 = -8.128743550630e+01; // 0xC054526557E4D2F2
const RA7: f64 = -9.814329344169e+00; // 0xC023A0EFC69AC25C
const SA1: f64 = 1.965127166743e+01; // 0x4033A6B9BD707687
const SA2: f64 = 1.376577541435e+02; // 0x4061350C526AE721
const SA3: f64 = 4.345658774752e+02; // 0x407B290DD58A1A71
const SA4: f64 = 6.453872717332e+02; // 0x40842B1921EC2868
const SA5: f64 = 4.290081400275e+02; // 0x407AD02157700314
const SA6: f64 = 1.086350055417e+02; // 0x405B28A3EE48AE2C
const SA7: f64 = 6.570249770319e+00; // 0x401A47EF8E484A93
const SA8: f64 = -6.042441521485e-02; // 0xBFAEEFF2EE749A62

// Coefficients for approximation to  erfc in [1/.35, 28]
const RB0: f64 = -9.864942924700e-03; // 0xBF84341239E86F4A
const RB1: f64 = -7.992832376805e-01; // 0xBFE993BA70C285DE
const RB2: f64 = -1.775795491775e+01; // 0xC031C209555F995A
const RB3: f64 = -1.606363848558e+02; // 0xC064145D43C5ED98
const RB4: f64 = -6.375664433683e+02; // 0xC083EC881375F228
const RB5: f64 = -1.025095131611e+03; // 0xC09004616A2E5992
const RB6: f64 = -4.835191916086e+02; // 0xC07E384E9BDC383F
const SB1: f64 = 3.033806074348e+01; // 0x403E568B261D5190
const SB2: f64 = 3.257925129965e+02; // 0x40745CAE221B9F0A
const SB3: f64 = 1.536729586084e+03; // 0x409802EB189D5118
const SB4: f64 = 3.199858219508e+03; // 0x40A8FFB7688C246A
const SB5: f64 = 2.553050406433e+03; // 0x40A3F219CEDF3BE6
const SB6: f64 = 4.745285412069e+02; // 0x407DA874E79FE763
const SB7: f64 = -2.244095244658e+01; // 0xC03670E242712D62

const ERX: f64 = 8.450629115104675e-1; // 0x3FEB0AC160000000
const TINY: f64 = 2.848094538889218e-306; // 0x0080000000000000
const SMALL: f64 = 1.0 / (1 << 28) as f64; // 2**-28

enum Interval {
    Tiny,
    Small,
    I084,
    I125,
    I1DIVIDE035,
    I6,
    Default,
}

fn get_interval(x: f64) -> Interval {
    if x <= TINY {
        return Interval::Tiny;
    }
    if x <= SMALL {
        return Interval::Small;
    }
    if x <= 0.84375 {
        return Interval::I084;
    }
    if x <= 1.25 {
        return Interval::I125;
    }
    if x <= 1.0 / 0.35 {
        return Interval::I1DIVIDE035;
    }
    if x <= 6.0 {
        return Interval::I6;
    }
    Interval::Default
}

pub fn erf(x: f64) -> f64 {
    let y = x.abs();

    if x == f64::INFINITY {
        return 1.0;
    } else if x == f64::NEG_INFINITY {
        return -1.0;
    } else if x.is_nan() {
        return f64::NAN;
    }

    // Special case for large negative values
    if x <= -5.0 {
        return -1.0; // erf(x) tends to -1 as x -> -∞
    }

    match get_interval(y) {
        // special cases
        Interval::Tiny => 0.125 * (8.0 * y + EFX8 * y) * x.signum(),
        Interval::Small => (y + EFX * y) * x.signum(), // avoid underflow
        Interval::I084 => {
            let z = y.powi(2);
            let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
            let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
            (y + y * (r / s)) * x.signum()
        }
        Interval::I125 => {
            let s = y - 1.0;
            let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
            let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
            (ERX + p / q) * x.signum()
        }
        Interval::I1DIVIDE035 => {
            let s = 1.0 / y.powi(2);
            let r = RA0
                + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
            let s2 = 1.0
                + s * (SA1
                    + s * (SA2
                        + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
            let z = (y.to_bits() & 0xffffffff00000000) as f64; // pseudo-single (20-bit) precision x
            let r2 = (-z * z - 0.5625).exp() * ((z - y) * (z + y) + r / s2).exp();
            (1.0 - r2 / y) * x.signum()
        }
        Interval::I6 => {
            let s = 1.0 / y.powi(2);
            let r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
            let s2 = 1.0
                + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
            let z = (y.to_bits() & 0xffffffff00000000) as f64; // pseudo-single (20-bit) precision x
            let r2 = (-z * z - 0.5625).exp() * ((z - y) * (z + y) + r / s2).exp();
            (1.0 - r2 / y) * x.signum()
        }
        _ => y.signum(),
    }
}


