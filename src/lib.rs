extern crate num;

use std::f64::consts::PI;
use std::f64::consts::SQRT_2;
use std::f64::{NAN, NEG_INFINITY, INFINITY};

// Coefficients for approximation to  erf in [0, 0.84375]
const efx: f64  = 1.28379167095512586316e-01;  // 0x3FC06EBA8214DB69
const efx8: f64 = 1.02703333676410069053e+00;  // 0x3FF06EBA8214DB69
const pp0: f64  = 1.28379167095512558561e-01;  // 0x3FC06EBA8214DB68
const pp1: f64  = -3.25042107247001499370e-01; // 0xBFD4CD7D691CB913
const pp2: f64  = -2.84817495755985104766e-02; // 0xBF9D2A51DBD7194F
const pp3: f64  = -5.77027029648944159157e-03; // 0xBF77A291236668E4
const pp4: f64  = -2.37630166566501626084e-05; // 0xBEF8EAD6120016AC
const qq1: f64  = 3.97917223959155352819e-01;  // 0x3FD97779CDDADC09
const qq2: f64  = 6.50222499887672944485e-02;  // 0x3FB0A54C5536CEBA
const qq3: f64  = 5.08130628187576562776e-03;  // 0x3F74D022C4D36B0F
const qq4: f64  = 1.32494738004321644526e-04;  // 0x3F215DC9221C1A10
const qq5: f64  = -3.96022827877536812320e-06; // 0xBED09C4342A26120

// Coefficients for approximation to  erf  in [0.84375, 1.25]
const pa0: f64 = -2.36211856075265944077e-03; // 0xBF6359B8BEF77538
const pa1: f64 = 4.14856118683748331666e-01;  // 0x3FDA8D00AD92B34D
const pa2: f64 = -3.72207876035701323847e-01; // 0xBFD7D240FBB8C3F1
const pa3: f64 = 3.18346619901161753674e-01;  // 0x3FD45FCA805120E4
const pa4: f64 = -1.10894694282396677476e-01; // 0xBFBC63983D3E28EC
const pa5: f64 = 3.54783043256182359371e-02;  // 0x3FA22A36599795EB
const pa6: f64 = -2.16637559486879084300e-03; // 0xBF61BF380A96073F
const qa1: f64 = 1.06420880400844228286e-01;  // 0x3FBB3E6618EEE323
const qa2: f64 = 5.40397917702171048937e-01;  // 0x3FE14AF092EB6F33
const qa3: f64 = 7.18286544141962662868e-02;  // 0x3FB2635CD99FE9A7
const qa4: f64 = 1.26171219808761642112e-01;  // 0x3FC02660E763351F
const qa5: f64 = 1.36370839120290507362e-02;  // 0x3F8BEDC26B51DD1C
const qa6: f64 = 1.19844998467991074170e-02;  // 0x3F888B545735151D

// Coefficients for approximation to  erfc in [1.25, 1/0.35]
const ra0: f64 = -9.86494403484714822705e-03; // 0xBF843412600D6435
const ra1: f64 = -6.93858572707181764372e-01; // 0xBFE63416E4BA7360
const ra2: f64 = -1.05586262253232909814e+01; // 0xC0251E0441B0E726
const ra3: f64 = -6.23753324503260060396e+01; // 0xC04F300AE4CBA38D
const ra4: f64 = -1.62396669462573470355e+02; // 0xC0644CB184282266
const ra5: f64 = -1.84605092906711035994e+02; // 0xC067135CEBCCABB2
const ra6: f64 = -8.12874355063065934246e+01; // 0xC054526557E4D2F2
const ra7: f64 = -9.81432934416914548592e+00; // 0xC023A0EFC69AC25C
const sa1: f64 = 1.96512716674392571292e+01;  // 0x4033A6B9BD707687
const sa2: f64 = 1.37657754143519042600e+02;  // 0x4061350C526AE721
const sa3: f64 = 4.34565877475229228821e+02;  // 0x407B290DD58A1A71
const sa4: f64 = 6.45387271733267880336e+02;  // 0x40842B1921EC2868
const sa5: f64 = 4.29008140027567833386e+02;  // 0x407AD02157700314
const sa6: f64 = 1.08635005541779435134e+02;  // 0x405B28A3EE48AE2C
const sa7: f64 = 6.57024977031928170135e+00;  // 0x401A47EF8E484A93
const sa8: f64 = -6.04244152148580987438e-02; // 0xBFAEEFF2EE749A62

// Coefficients for approximation to  erfc in [1/.35, 28]
const rb0: f64 = -9.86494292470009928597e-03; // 0xBF84341239E86F4A
const rb1: f64 = -7.99283237680523006574e-01; // 0xBFE993BA70C285DE
const rb2: f64 = -1.77579549177547519889e+01; // 0xC031C209555F995A
const rb3: f64 = -1.60636384855821916062e+02; // 0xC064145D43C5ED98
const rb4: f64 = -6.37566443368389627722e+02; // 0xC083EC881375F228
const rb5: f64 = -1.02509513161107724954e+03; // 0xC09004616A2E5992
const rb6: f64 = -4.83519191608651397019e+02; // 0xC07E384E9BDC383F
const sb1: f64 = 3.03380607434824582924e+01;  // 0x403E568B261D5190
const sb2: f64 = 3.25792512996573918826e+02;  // 0x40745CAE221B9F0A
const sb3: f64 = 1.53672958608443695994e+03;  // 0x409802EB189D5118
const sb4: f64 = 3.19985821950859553908e+03;  // 0x40A8FFB7688C246A
const sb5: f64 = 2.55305040643316442583e+03;  // 0x40A3F219CEDF3BE6
const sb6: f64 = 4.74528541206955367215e+02;  // 0x407DA874E79FE763
const sb7: f64 = -2.24409524465858183362e+01; // 0xC03670E242712D62

const K: f64 = 1.0/0.35;

const erx: f64       = 8.45062911510467529297e-01; // 0x3FEB0AC160000000
const very_tiny: f64 = 2.848094538889218e-306; // 0x0080000000000000
const small: f64     = 1.0 / (1 << 28) as f64;        // 2**-28

pub fn erf(x: f64) -> f64 {
    let y = x.abs();

    match y {
        // special cases
        NAN => NAN,
        NEG_INFINITY => -1.0,
        INFINITY => 1.0,
        NEG_INFINITY...very_tiny =>  0.125 * (8.0*y + efx8*y) * x.signum(),
        very_tiny...small =>  (y + efx * y) * x.signum(), // avoid underflow
        small...0.84375 => {
            let z = y.powi(2);
            let r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            let s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            (y + y*(r/s)) * x.signum()
        },
        0.84375...1.25 =>  {
    		let s = y - 1.0;
    		let P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
		    let Q = 1.0 + s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))));
            (erx + P/Q) * x.signum() 
        },
        1.25...K => {
            let s = 1.0 / y.powi(2);
    		let R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
    		let S = 1.0 + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s *(sa7 + s * sa8)))))));
            let z = (y.to_bits() & 0xffffffff00000000) as f64; // pseudo-single (20-bit) precision x
    	    let r = (-z * z - 0.5625).exp() * ((z - y) * (z + y) + R / S).exp();
            (1.0 - r/y) * x.signum()
            //
    	},
        K...6.0 => { // |x| >= 1 / 0.35  ~ 2.857143
            let s = 1.0 / y.powi(2);
    		let R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
    		let S = 1.0 + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
            let z = (y.to_bits() & 0xffffffff00000000) as f64; // pseudo-single (20-bit) precision x
    	    let r = (-z * z - 0.5625).exp() * ((z - y) * (z + y) + R/S).exp();
            (1.0 - r/y) * x.signum()
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
    fn test_cummulative_distrib() {
        let NCDF_0       = 0.5;
        let NCDF_1       = 0.8413447460685429;
        let NCDF_MINUS_1 = 0.15865525393145707;
        assert_eq!(cummulative_distrib( 0.0, 0.0, 1.0), NCDF_0);
        assert_eq!(cummulative_distrib( 1.0, 0.0, 1.0), NCDF_1);
        assert_eq!(cummulative_distrib(-1.0, 0.0, 1.0), NCDF_MINUS_1);
    }
    // TODO test_erf
}
