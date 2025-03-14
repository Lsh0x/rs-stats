# rs-stats

[![Rust](https://img.shields.io/badge/rust-1.56%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)](https://crates.io/crates/rs-stats)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/lsh0x/rs-stats/actions)
[![GitHub last commit](https://img.shields.io/github/last-commit/lsh0x/rs-stats)](https://github.com/lsh0x/rs-stats/commits/main)
[![CI](https://github.com/lsh0x/rs-stats/workflows/CI/badge.svg)](https://github.com/lsh0x/rs-stats/actions)
[![Codecov](https://codecov.io/gh/lsh0x/rs-stats/branch/main/graph/badge.svg)](https://codecov.io/gh/lsh0x/rs-stats)
[![Docs](https://docs.rs/rs-stats/badge.svg)](https://docs.rs/rs-stats)
[![Crates.io](https://img.shields.io/crates/v/rs-stats.svg)](https://crates.io/crates/rs-stats)
[![crates.io](https://img.shields.io/crates/d/rs-stats)](https://crates.io/crates/rs-stats)

A comprehensive statistical library written in Rust, providing powerful tools for probability, distributions, and hypothesis testing.


rs-stats offers a broad range of statistical functionality implemented in pure Rust. It's designed to be intuitive, efficient, and reliable for both simple and complex statistical analysis. The library aims to provide a comprehensive set of tools for data scientists, researchers, and developers working with statistical models.

## Features

- **Probability Functions**
  - Error functions (erf, erfc)
  - Cumulative distribution functions
  - Probability density functions
  - Z-scores
  - Basic statistics (mean, variance, standard deviation, standard error)

- **Statistical Distributions**
  - Normal (Gaussian) distribution
  - Binomial distribution
  - Exponential distribution
  - Poisson distribution
  - Uniform distribution

- **Regression Analysis**
  - Linear Regression (fit, predict, confidence intervals)
  - Multiple Linear Regression (multiple predictor variables)
  - Model statistics (R², adjusted R², standard error)
  - Model persistence (save/load models in JSON or binary format)

- **Hypothesis Testing**
  - ANOVA (Analysis of Variance)
  - Chi-square tests (independence and goodness of fit)
  - T-tests (one-sample, two-sample, paired)

## Installation

Add rs-stats to your `Cargo.toml`:

```toml
[dependencies]
rs-stats = "1.2.0"
```

Or use cargo add:

```bash
cargo add rs-stats
```

## Usage Examples

### Basic Statistical Functions

```rust
use rs_stats::prob::{average, variance, population_std_dev, std_err};

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let mean = average(&data);
    let var = variance(&data);
    let std_dev = population_std_dev(&data);
    let std_error = std_err(&data);
    
    println!("Mean: {}", mean);
    println!("Variance: {}", var);
    println!("Standard Deviation: {}", std_dev);
    println!("Standard Error: {}", std_error);
}
```

### Working with Distributions

```rust
use rs_stats::distributions::normal_distribution::{normal_pdf, normal_cdf, normal_inverse_cdf};

fn main() {
    // Standard normal distribution (mean=0, std_dev=1)
    let x = 1.96;
    
    // Probability density at x
    let density = normal_pdf(x, 0.0, 1.0);
    println!("PDF at {}: {}", x, density);
    
    // Cumulative probability P(X ≤ x)
    let cumulative = normal_cdf(x, 0.0, 1.0);
    println!("CDF at {}: {}", x, cumulative);
    
    // Inverse CDF (quantile function)
    let p = 0.975;
    let quantile = normal_inverse_cdf(p, 0.0, 1.0);
    println!("{}th percentile: {}", p * 100.0, quantile);
}
```

### Hypothesis Testing

```rust
use rs_stats::hypothesis_tests::t_test::{one_sample_t_test, two_sample_t_test};
use rs_stats::hypothesis_tests::chi_square_test::{chi_square_goodness_of_fit, chi_square_independence};
use rs_stats::hypothesis_tests::anova::one_way_anova;

fn main() {
    // One-sample t-test
    let sample = vec![5.1, 5.2, 4.9, 5.0, 5.3];
    let result = one_sample_t_test(&sample, 5.0);
    println!("One-sample t-test p-value: {}", result.p_value);
    
    // Two-sample t-test
    let sample1 = vec![5.1, 5.2, 4.9, 5.0, 5.3];
    let sample2 = vec![4.8, 4.9, 5.0, 4.7, 4.9];
    let result = two_sample_t_test(&sample1, &sample2);
    println!("Two-sample t-test p-value: {}", result.p_value);
    
    // ANOVA
    let groups = vec![
        vec![5.1, 5.2, 4.9, 5.0, 5.3],
        vec![4.8, 4.9, 5.0, 4.7, 4.9],
        vec![5.2, 5.3, 5.1, 5.4, 5.2],
    ];
    let result = one_way_anova(&groups);
    println!("ANOVA p-value: {}", result.p_value);
    
    // Chi-square test of independence
    let observed = vec![
        vec![45, 55],
        vec![60, 40],
    ];
    let result = chi_square_independence(&observed);
    println!("Chi-square independence test p-value: {}", result.p_value);
}
```

### Regression Analysis

```rust
use rs_stats::regression::linear_regression::LinearRegression;
use rs_stats::regression::multiple_linear_regression::MultipleLinearRegression;

fn main() {
    // Simple Linear Regression
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    
    println!("Slope: {}", model.slope);
    println!("Intercept: {}", model.intercept);
    println!("R-squared: {}", model.r_squared);
    
    // Predict new values
    let prediction = model.predict(6.0);
    println!("Prediction for x=6: {}", prediction);
    
    // Calculate confidence interval (95%)
    if let Some((lower, upper)) = model.confidence_interval(6.0, 0.95) {
        println!("95% confidence interval: ({}, {})", lower, upper);
    }
    
    // Multiple Linear Regression
    let x_multi = vec![
        vec![1.0, 2.0], // observation 1: x1.2.0, x2=2.0
        vec![2.0, 1.0], // observation 2: x1=2.0, x2=1.0
        vec![3.0, 3.0], // observation 3: x1=3.0, x2=3.0
        vec![4.0, 2.0], // observation 4: x1=4.0, x2=2.0
    ];
    let y_multi = vec![9.0, 8.0, 16.0, 15.0];
    
    let mut multi_model = MultipleLinearRegression::new();
    multi_model.fit(&x_multi, &y_multi).unwrap();
    
    println!("Coefficients: {:?}", multi_model.coefficients);
    println!("R-squared: {}", multi_model.r_squared);
    println!("Adjusted R-squared: {}", multi_model.adjusted_r_squared);
    
    // Predict with multiple variables
    let new_observation = vec![5.0, 4.0];
    let prediction = multi_model.predict(&new_observation);
    println!("Prediction for new observation: {}", prediction);
    
    // Save model to file
    multi_model.save("model.json").unwrap();
    
    // Load model from file
    let loaded_model = MultipleLinearRegression::load("model.json").unwrap();
}
```

### Decision Trees

```rust
use rs_stats::regression::decision_tree::{DecisionTree, TreeType, SplitCriterion};

// Example 1: Regression Tree for Patient Recovery Time Prediction
let mut recovery_time_tree = DecisionTree::<f64, f64>::new(
    TreeType::Regression,
    SplitCriterion::Mse,
    5,   // max_depth
    2,   // min_samples_split
    1    // min_samples_leaf
);

// Training data: [age, treatment_intensity, bmi, comorbidity_score, initial_severity]
let patient_features = vec![
    vec![45.0, 3.0, 28.5, 2.0, 7.0],  // Patient 1: 45 years, treatment intensity 3, BMI 28.5, etc.
    vec![62.0, 4.0, 31.2, 3.0, 8.0],  // Patient 2
    vec![38.0, 2.0, 24.3, 1.0, 5.0],  // Patient 3
    // ... more patients
];
let recovery_days = vec![14.0, 28.0, 10.0];  // Recovery time in days

// Train the model to predict recovery time
recovery_time_tree.fit(&patient_features, &recovery_days);

// Make predictions for a new patient
let new_patient = vec![
    vec![55.0, 3.0, 27.0, 2.0, 6.0],  // New patient characteristics
];
let predicted_recovery_days = recovery_time_tree.predict(&new_patient);

// Example 2: Classification Tree for Diabetes Risk Assessment
let mut diabetes_risk_tree = DecisionTree::<u8, f64>::new(
    TreeType::Classification,
    SplitCriterion::Gini,
    4,   // max_depth
    2,   // min_samples_split
    1    // min_samples_leaf
);

// Training data: [glucose_level, bmi, blood_pressure, age, family_history]
let medical_features = vec![
    vec![85.0, 22.0, 120.0, 35.0, 0.0],  // Patient 1: glucose 85 mg/dL, BMI 22, BP 120, etc.
    vec![140.0, 31.0, 145.0, 52.0, 1.0],  // Patient 2
    vec![165.0, 34.0, 155.0, 48.0, 1.0],  // Patient 3
    // ... more patients
];
let diabetes_status = vec![0, 1, 1];  // 0: No diabetes, 1: Diabetes

// Train the classifier
diabetes_risk_tree.fit(&medical_features, &diabetes_status);

// Print tree structure and summary
println!("Tree Structure:\n{}", diabetes_risk_tree.tree_structure());
println!("Tree Summary:\n{}", diabetes_risk_tree.summary());

// Feature importance - which medical measurements are most predictive
let importance = diabetes_risk_tree.feature_importances();
println!("Feature Importance: {:?}", importance);
```

The Decision Tree implementation supports:
- Both regression and classification tasks
- Multiple split criteria (MSE, MAE for regression; Gini, Entropy for classification)
- Generic types with appropriate trait bounds
- Parallel processing for optimal performance
- Tree visualization and interpretation tools
- Feature importance calculation

## Documentation

For detailed API documentation, run:

```bash
cargo doc --open
```

## Testing

The library includes a comprehensive test suite. Run the tests with:

```bash
cargo test
```

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Submit a pull request

Before submitting your PR, please make sure:
- All tests pass
- Code follows the project's style and conventions
- New features include appropriate documentation and tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Rust community for their excellent documentation and support
- Contributors to the project
- Various statistical references and research papers that informed the implementations
