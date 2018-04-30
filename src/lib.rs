extern crate num;

use std::f64::consts::PI;
use std::f64::consts::SQRT_2;
use std::f64::NAN;

// Coefficients for approximation to  erf in [0, 0.84375]
const EFX: f64  = 1.28379167095512586316e-01;  // 0x3FC06EBA8214DB69
const EFX8: f64 = 1.02703333676410069053e+00;  // 0x3FF06EBA8214DB69
const PP0: f64  = 1.28379167095512558561e-01;  // 0x3FC06EBA8214DB68
const PP1: f64  = -3.25042107247001499370e-01; // 0xBFD4CD7D691CB913
const PP2: f64  = -2.84817495755985104766e-02; // 0xBF9D2A51DBD7194F
const PP3: f64  = -5.77027029648944159157e-03; // 0xBF77A291236668E4
const PP4: f64  = -2.37630166566501626084e-05; // 0xBEF8EAD6120016AC
const QQ1: f64  = 3.97917223959155352819e-01;  // 0x3FD97779CDDADC09
const QQ2: f64  = 6.50222499887672944485e-02;  // 0x3FB0A54C5536CEBA
const QQ3: f64  = 5.08130628187576562776e-03;  // 0x3F74D022C4D36B0F
const QQ4: f64  = 1.32494738004321644526e-04;  // 0x3F215DC9221C1A10
const QQ5: f64  = -3.96022827877536812320e-06; // 0xBED09C4342A26120

// Coefficients for approximation to  erf  in [0.84375, 1.25]
const PA0: f64 = -2.36211856075265944077e-03; // 0xBF6359B8BEF77538
const PA1: f64 = 4.14856118683748331666e-01;  // 0x3FDA8D00AD92B34D
const PA2: f64 = -3.72207876035701323847e-01; // 0xBFD7D240FBB8C3F1
const PA3: f64 = 3.18346619901161753674e-01;  // 0x3FD45FCA805120E4
const PA4: f64 = -1.10894694282396677476e-01; // 0xBFBC63983D3E28EC
const PA5: f64 = 3.54783043256182359371e-02;  // 0x3FA22A36599795EB
const PA6: f64 = -2.16637559486879084300e-03; // 0xBF61BF380A96073F
const QA1: f64 = 1.06420880400844228286e-01;  // 0x3FBB3E6618EEE323
const QA2: f64 = 5.40397917702171048937e-01;  // 0x3FE14AF092EB6F33
const QA3: f64 = 7.18286544141962662868e-02;  // 0x3FB2635CD99FE9A7
const QA4: f64 = 1.26171219808761642112e-01;  // 0x3FC02660E763351F
const QA5: f64 = 1.36370839120290507362e-02;  // 0x3F8BEDC26B51DD1C
const QA6: f64 = 1.19844998467991074170e-02;  // 0x3F888B545735151D

// Coefficients for approximation to  erfc in [1.25, 1/0.35]
const RA0: f64 = -9.86494403484714822705e-03; // 0xBF843412600D6435
const RA1: f64 = -6.93858572707181764372e-01; // 0xBFE63416E4BA7360
const RA2: f64 = -1.05586262253232909814e+01; // 0xC0251E0441B0E726
const RA3: f64 = -6.23753324503260060396e+01; // 0xC04F300AE4CBA38D
const RA4: f64 = -1.62396669462573470355e+02; // 0xC0644CB184282266
const RA5: f64 = -1.84605092906711035994e+02; // 0xC067135CEBCCABB2
const RA6: f64 = -8.12874355063065934246e+01; // 0xC054526557E4D2F2
const RA7: f64 = -9.81432934416914548592e+00; // 0xC023A0EFC69AC25C
const SA1: f64 = 1.96512716674392571292e+01;  // 0x4033A6B9BD707687
const SA2: f64 = 1.37657754143519042600e+02;  // 0x4061350C526AE721
const SA3: f64 = 4.34565877475229228821e+02;  // 0x407B290DD58A1A71
const SA4: f64 = 6.45387271733267880336e+02;  // 0x40842B1921EC2868
const SA5: f64 = 4.29008140027567833386e+02;  // 0x407AD02157700314
const SA6: f64 = 1.08635005541779435134e+02;  // 0x405B28A3EE48AE2C
const SA7: f64 = 6.57024977031928170135e+00;  // 0x401A47EF8E484A93
const SA8: f64 = -6.04244152148580987438e-02; // 0xBFAEEFF2EE749A62

// Coefficients for approximation to  erfc in [1/.35, 28]
const RB0: f64 = -9.86494292470009928597e-03; // 0xBF84341239E86F4A
const RB1: f64 = -7.99283237680523006574e-01; // 0xBFE993BA70C285DE
const RB2: f64 = -1.77579549177547519889e+01; // 0xC031C209555F995A
const RB3: f64 = -1.60636384855821916062e+02; // 0xC064145D43C5ED98
const RB4: f64 = -6.37566443368389627722e+02; // 0xC083EC881375F228
const RB5: f64 = -1.02509513161107724954e+03; // 0xC09004616A2E5992
const RB6: f64 = -4.83519191608651397019e+02; // 0xC07E384E9BDC383F
const SB1: f64 = 3.03380607434824582924e+01;  // 0x403E568B261D5190
const SB2: f64 = 3.25792512996573918826e+02;  // 0x40745CAE221B9F0A
const SB3: f64 = 1.53672958608443695994e+03;  // 0x409802EB189D5118
const SB4: f64 = 3.19985821950859553908e+03;  // 0x40A8FFB7688C246A
const SB5: f64 = 2.55305040643316442583e+03;  // 0x40A3F219CEDF3BE6
const SB6: f64 = 4.74528541206955367215e+02;  // 0x407DA874E79FE763
const SB7: f64 = -2.24409524465858183362e+01; // 0xC03670E242712D62


const ERX: f64       = 8.45062911510467529297e-01; // 0x3FEB0AC160000000
const TINY: f64      = 2.848094538889218e-306; // 0x0080000000000000
const SMALL: f64     = 1.0 / (1 << 28) as f64;        // 2**-28

enum Interval {
    NAN,
    NEGINFINITY,
    INFINITY,
    TINY,
    SMALL,
    I084,
    I125,
    I1DIVIDE035,
    I6,
    DEFAULT,
}


fn get_interval(x: f64) -> Interval {
    if x.is_nan() {
        return Interval::NAN
    }
    if x.is_infinite() {
        match x.is_sign_positive() {
            true => return Interval::INFINITY,
            false => return Interval::NEGINFINITY
        }
    }
    if x <= TINY {
        return Interval::TINY;
    }
    if x <= SMALL {
        return Interval::SMALL;
    }
    if x <= 0.84375 {
        return Interval::I084;
    }
    if x <= 1.25 {
        return Interval::I125;
    }
    if x <= 1.0/0.35 {
        return Interval::I1DIVIDE035;
    }
    if x <= 6.0 {
        return Interval::I6;
    }
    return Interval::DEFAULT;
} 

pub fn erf(x: f64) -> f64 {
    let y = x.abs();

    match get_interval(y) {
        // special cases
        Interval::NAN => NAN,
        Interval::NEGINFINITY => -1.0,
        Interval::INFINITY => 1.0,
        Interval::TINY =>  0.125 * (8.0*y + EFX8*y) * x.signum(),
        Interval::SMALL =>  (y + EFX * y) * x.signum(), // avoid underflow
        Interval::I084 => {
            let z = y.powi(2);
            let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
            let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
            (y + y*(r/s)) * x.signum()
        },
        Interval::I125 =>  {
    		let s = y - 1.0;
    		let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
		    let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
            (ERX + p/q) * x.signum() 
        },
        Interval::I1DIVIDE035 => {
            let s = 1.0 / y.powi(2);
    		let r  = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
    		let s2 = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s *(SA7 + s * SA8)))))));
            let z = (y.to_bits() & 0xffffffff00000000) as f64; // pseudo-single (20-bit) precision x
    	    let r2 = (-z * z - 0.5625).exp() * ((z - y) * (z + y) + r / s2).exp();
            (1.0 - r2/y) * x.signum()
    	},
        Interval::I6 => { // |x| >= 1 / 0.35  ~ 2.857143
            let s = 1.0 / y.powi(2);
    		let r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
    		let s2 = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
            let z = (y.to_bits() & 0xffffffff00000000) as f64; // pseudo-single (20-bit) precision x
    	    let r2 = (-z * z - 0.5625).exp() * ((z - y) * (z + y) + r/s2).exp();
            (1.0 - r2/y) * x.signum()
        },
        _ => y.signum()
    }
}

/// average gets the number expressing the central or typical value in a set of data
pub fn average<T: num::ToPrimitive >(t: &[T]) -> Option<f64>  {
    if t.len() == 0 {
        return None
    }
    Some(t.iter().map(|x| x.to_f64().unwrap()).sum::<f64>() / t.len() as f64)
}

/// variance us the mean of the sum of all square deviation
pub fn variance<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    match average(t) {
        Some(avg) => {
            let len: f64 = t.len() as f64;
            Some(&t.iter().map(|x| (x.to_f64().unwrap() - avg ).powi(2)).sum::<f64>() / len )
        },
        None => None,
    }
}

/// std_dev return the standard deviation, the square root of the variance
pub fn std_dev<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    match variance(t) {
        Some(x) => Some(x.sqrt()),
        None => None,
    }
}

/// std_err is the standard error, represnting the standard deviation of its distribution
pub fn std_err<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    match std_dev(t) {
        Some(std) => Some(std / (t.len() as f64).sqrt()),
        None => None
    }
}

/// the zscore represente the distance from the mean in stddev
pub fn zscore(x: f64, avg: f64, stddev: f64) -> f64 {
   (x - avg) / stddev 
}

/// probability_density normalize x using the mean and the standard deviation and return the PDF
/// https://en.wikipedia.org/wiki/Probability_density_function
pub fn probability_density(x: f64, avg: f64, stddev: f64) -> f64 {
    (zscore(x, avg, stddev).powi(2) / -2.0).exp() / (stddev * (PI * 2.0).sqrt())
}

/// normal_probability_density return the PDF with z already normalized
/// https://en.wikipedia.org/wiki/Probability_density_function
pub fn normal_probability_density(z: f64) -> f64 {
    (z.powi(2) / -2.0).exp() / (PI * 2.0).sqrt()
}

/// CDF return the CDF using the mean and the standard deviation given
/// https://en.wikipedia.org/wiki/Cumulative_distribution_function#Definition
pub fn cummulative_distrib(x: f64, avg: f64, stddev: f64) -> f64 {
   (1.0 + erf(zscore(x, avg, stddev) / SQRT_2)) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average() {
        assert_eq!(average(&[-2, -3, -4]), Some(-3.0));
        assert_eq!(average(&[1.0, 1.0, 1.0]), Some(1.0));
        assert_eq!(average(&[0, 0, 0]), Some(0.0));
        let mut vec = vec!();
        vec.push(42);
        vec.clear();
        assert_eq!(average(&vec), None);
    }

    #[test]
    fn test_variance() {
        assert_eq!(variance(&[3, 5]), Some(1.0));
        assert_eq!(variance(&[3.0, 5.0]), Some(1.0));
        assert_eq!(variance(&[1, 1, 1]), Some(0.0));
        let mut vec = vec!();
        vec.push(42);
        vec.clear();
        assert_eq!(variance(&vec), None);
    }

    #[test]
    fn test_std_dev() {
        assert_eq!(std_dev(&[3, 5]), Some(1.0));
        assert_eq!(std_dev(&[3.0, 5.0]), Some(1.0));
        assert_eq!(std_dev(&[1, 1, 1]), Some(0.0));
        let mut vec = vec!();
        vec.push(42);
        vec.clear();
        assert_eq!(std_dev(&vec), None);
    }

    #[test]
    fn test_std_err() {
        assert_eq!(std_err(&[1, 2, 2, 1]), Some(0.25));
        assert_eq!(std_err(&[100.0, 150.0, 150.0, 100.0]), Some(12.5));
        let mut vec = vec!();
        vec.push(42);
        vec.clear();
        assert_eq!(std_err(&vec), None);
    }

    #[test]
    fn test_zscore() {
        assert_eq!(zscore(10.0, 10.0, 1.0), 0.0);
        assert_eq!(zscore(15.0, 10.0, 1.0), 5.0);
    }

    #[test]
    fn test_probability_density_function() {
        let  ret = 0.3989422804014327;
        let  ret2 = 0.24197072451914337;

        assert_eq!(probability_density(0.0, 0.0, 1.0), ret);
        assert_eq!(probability_density(-1.0, 0.0, 1.0), ret2);
        assert_eq!(probability_density(1.0, 0.0, 1.0), ret2);
    }

    #[test]
    fn test_normal_probability_density_function() {
        let  ret = 0.3989422804014327;
        let  ret2 = 0.24197072451914337;

        assert_eq!(normal_probability_density(0.0), ret);
        assert_eq!(normal_probability_density(-1.0), ret2);
        assert_eq!(normal_probability_density(1.0), ret2);
    }

    #[test]
    fn test_erf() {
        let ret = 0.8427007929497149;
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(1.0), ret);
        assert_eq!(erf(-1.0), -ret);
    }

    #[test]
    fn test_cummulative_distrib() {
        let ret_1 = 0.5;
        let ret_2 = 0.8413447460685429;
        let ret_3 =  0.15865525393145707;
        assert_eq!(cummulative_distrib( 0.0, 0.0, 1.0), ret_1);
        assert_eq!(cummulative_distrib( 1.0, 0.0, 1.0), ret_2);
        assert_eq!(cummulative_distrib(-1.0, 0.0, 1.0), ret_3);
    }
    // TODO test_erf
}
