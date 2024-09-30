# rs-stats
Rust setup for new project

[![GitHub last commit](https://img.shields.io/github/last-commit/lsh0x/rs-stats)](https://github.com/lsh0x/rs-stats/commits/main)
[![CI](https://github.com/lsh0x/rs-stats/workflows/CI/badge.svg)](https://github.com/lsh0x/rs-stats/actions)
[![Codecov](https://codecov.io/gh/lsh0x/rs-stats/branch/main/graph/badge.svg)](https://codecov.io/gh/lsh0x/rs-stats)
[![Docs](https://docs.rs/rs-stats/badge.svg)](https://docs.rs/rs-stats)
[![Crates.io](https://img.shields.io/crates/v/rs-stats.svg)](https://crates.io/crates/rs-stats)
[![crates.io](https://img.shields.io/crates/d/rs-stats)](https://crates.io/crates/rs-stats)


rs-stats: A Rust Statistics Library

This library provides a collection of statistical functions implemented in Rust. It aims to offer a simple and efficient way to perform common statistical calculations.

Features:

- Error Functions:
    - erf(x: f64) -> f64: Calculates the error function of a given value.
    - erfc(x: f64) -> f64: Calculates the complementary error function of a given value.

- Normal Distribution:
    - normal_cummulative_distrib(z: f64) -> f64: Calculates the cumulative distribution function (CDF) of the standard normal distribution.
    - normal_probability_density(z: f64) -> f64: Calculates the probability density function (PDF) of the standard normal distribution.

- Descriptive Statistics:
    - average(data: &[f64]) -> f64: Calculates the average (mean) of a dataset.
    - stddev(data: &[f64]) -> f64: Calculates the population standard deviation of a dataset.
    - variance(data: &[f64]) -> f64: Calculates the population variance of a dataset.

- Other:
    - z_score(x: f64, avg: f64, stddev: f64) -> f64: Calculates the z-score of a value given the mean and standard deviation.
    - cummulative_distrib(x: f64, avg: f64, stddev: f64) -> f64: Calculates the cumulative distribution function (CDF) of a normal distribution with a given mean and standard deviation.
    - probability_density(x: f64, avg: f64, stddev: f64) -> f64: Calculates the probability density function (PDF) of a normal distribution with a given mean and standard deviation.
    - std_err(data: &[f64]) -> f64: Calculates the standard error of the mean.

Usage:

Add this to your Cargo.toml:

[dependencies]
rs-stats = "0.3.0" # Replace with the actual version

Then, in your Rust code:

use rs_stats::average; // Import the function

```
fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let avg = average(&data);
    println!("Average: {}", avg);
}
```

Contributing:

Contributions are welcome! Feel free to open issues or submit pull requests.

License:

This project is licensed under the MIT License - see the LICENSE   
 file for details.