pub mod average;
pub mod variance;
pub mod stddev;
pub mod zscore;
pub mod erf;

use crate::average::average;
use crate::variance::variance;
use crate::stddev::std_dev;
use crate::zscore::z_score;
use crate::erf::erf;

use std::f64::consts::PI;
use std::f64::consts::SQRT_2;

/// erfc returns the error function integrated between x and infinity
/// called complementary error function
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// std_err is the standard error, represnting the standard deviation of its distribution
pub fn std_err<T: num::ToPrimitive>(t: &[T]) -> Option<f64> {
    std_dev(t).map(|std| std / (t.len() as f64).sqrt())
}


/// probability_density normalize x using the mean and the standard deviation and return the PDF
/// https://en.wikipedia.org/wiki/Probability_density_function
pub fn probability_density(x: f64, avg: f64, stddev: f64) -> f64 {
    (z_score(x, avg, stddev).powi(2) / -2.0).exp() / (stddev * (PI * 2.0).sqrt())
}

/// normal_probability_density return the PDF with z already normalized
/// https://en.wikipedia.org/wiki/Probability_density_function
pub fn normal_probability_density(z: f64) -> f64 {
    (z.powi(2) / -2.0).exp() / (PI * 2.0).sqrt()
}

/// CDF return the CDF using the mean and the standard deviation given
/// https://en.wikipedia.org/wiki/Cumulative_distribution_function#Definition
pub fn cummulative_distrib(x: f64, avg: f64, stddev: f64) -> f64 {
    (1.0 + erf(z_score(x, avg, stddev) / SQRT_2)) / 2.0
}

/// CDF return the CDF for the zscore given
/// https://en.wikipedia.org/wiki/Cumulative_distribution_function#Definition
pub fn normal_cummulative_distrib(z: f64) -> f64 {
    (1.0 + erf(z / SQRT_2)) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_err() {
        assert_eq!(std_err(&[1, 2, 2, 1]), Some(0.25));
        assert_eq!(std_err(&[100.0, 150.0, 150.0, 100.0]), Some(12.5));
        let mut vec = vec![];
        vec.push(42);
        vec.clear();
        assert_eq!(std_err(&vec), None);
    }

    #[test]
    fn test_probability_density_function() {
        let ret = 0.3989422804014327;
        let ret2 = 0.24197072451914337;

        assert_eq!(probability_density(0.0, 0.0, 1.0), ret);
        assert_eq!(probability_density(-1.0, 0.0, 1.0), ret2);
        assert_eq!(probability_density(1.0, 0.0, 1.0), ret2);
    }

    #[test]
    fn test_normal_probability_density_function() {
        let ret = 0.3989422804014327;
        let ret2 = 0.24197072451914337;

        assert_eq!(normal_probability_density(0.0), ret);
        assert_eq!(normal_probability_density(-1.0), ret2);
        assert_eq!(normal_probability_density(1.0), ret2);
    }

    #[test]
    fn test_erf() {
        let ret = 0.8427007929497156;
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(1.0), ret);
        assert_eq!(erf(-1.0), -ret);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn test_erfc() {
        let ret = 0.8427007929497156;
        assert_eq!(erfc(0.0), 1.0);
        assert_eq!(erfc(1.0), 1.0 - ret);
        assert_eq!(erfc(-1.0), 1.0 + ret);
    }

    #[test]
    fn test_cummulative_distrib() {
        let ret_1 = 0.5;
        let ret_2 = 0.8413447460685398;
        let ret_3 = 0.15865525393146018;
        assert_eq!(cummulative_distrib(0.0, 0.0, 1.0), ret_1);
        assert_eq!(cummulative_distrib(1.0, 0.0, 1.0), ret_2);
        assert_eq!(cummulative_distrib(-1.0, 0.0, 1.0), ret_3);
    }

    #[test]
    fn test_normal_cummulative_distrib() {
        let ret_1 = 0.5;
        let ret_2 = 0.8413447460685398;
        let ret_3 = 0.15865525393146018;
        assert_eq!(normal_cummulative_distrib(z_score(0.0, 0.0, 1.0)), ret_1);
        assert_eq!(normal_cummulative_distrib(z_score(1.0, 0.0, 1.0)), ret_2);
        assert_eq!(normal_cummulative_distrib(z_score(-1.0, 0.0, 1.0)), ret_3);
    }
}
