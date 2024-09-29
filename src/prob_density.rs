use crate::zcore;

/// probability_density normalize x using the mean and the standard deviation and return the PDF
/// https://en.wikipedia.org/wiki/Probability_density_function
#[inline]
pub fn probability_density(x: f64, avg: f64, stddev: f64) -> f64 {
    (zscore(x, avg, stddev).powi(2) / -2.0).exp() / (stddev * (PI * 2.0).sqrt())
}
