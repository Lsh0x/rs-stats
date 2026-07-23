// src/regression/linear_regression.rs

use crate::error::{StatsError, StatsResult};
use num_traits::{Float, NumCast};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
#[cfg(feature = "serde")]
use std::fs::File;
#[cfg(feature = "serde")]
use std::io::{self};
#[cfg(feature = "serde")]
use std::path::Path;

/// Linear regression model that fits a line to data points.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LinearRegression<T = f64>
where
    T: Float + Debug + Default,
{
    /// Slope of the regression line (coefficient of x)
    pub slope: T,
    /// Y-intercept of the regression line
    pub intercept: T,
    /// Coefficient of determination (R²) - goodness of fit
    pub r_squared: T,
    /// Standard error of the estimate
    pub standard_error: T,
    /// Number of data points used for regression
    pub n: usize,
    /// Mean of the fitted X values (needed for confidence intervals).
    /// `serde(default)` keeps pre-v3.1 saved models loadable.
    #[cfg_attr(feature = "serde", serde(default))]
    pub x_mean: T,
    /// Sum of squared deviations of X around its mean, `Σ(xᵢ−x̄)²`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sum_xx: T,
}

impl<T> Default for LinearRegression<T>
where
    T: Float + Debug + Default + NumCast,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LinearRegression<T>
where
    T: Float + Debug + Default + NumCast,
{
    /// Create a new linear regression model without fitting any data
    pub fn new() -> Self {
        Self {
            slope: T::zero(),
            intercept: T::zero(),
            r_squared: T::zero(),
            standard_error: T::zero(),
            n: 0,
            x_mean: T::zero(),
            sum_xx: T::zero(),
        }
    }

    /// Fit a linear model to the provided x and y data points
    ///
    /// # Arguments
    /// * `x_values` - Independent variable values
    /// * `y_values` - Dependent variable values (observations)
    ///
    /// # Returns
    /// * `StatsResult<()>` - Ok if successful, Err with StatsError if the inputs are invalid
    ///
    /// # Errors
    /// Returns `StatsError::DimensionMismatch` if X and Y arrays have different lengths.
    /// Returns `StatsError::EmptyData` if the input arrays are empty.
    /// Returns `StatsError::ConversionError` if value conversion fails.
    /// Returns `StatsError::InvalidParameter` if there's no variance in X values.
    pub fn fit<U, V>(&mut self, x_values: &[U], y_values: &[V]) -> StatsResult<()>
    where
        U: NumCast + Copy,
        V: NumCast + Copy,
    {
        // Validate inputs
        if x_values.len() != y_values.len() {
            return Err(StatsError::dimension_mismatch(format!(
                "X and Y arrays must have the same length (got {} and {})",
                x_values.len(),
                y_values.len()
            )));
        }

        if x_values.is_empty() {
            return Err(StatsError::empty_data(
                "Cannot fit regression with empty arrays",
            ));
        }

        let n = x_values.len();
        self.n = n;

        // Convert input arrays to T type
        let x_cast: Vec<T> = x_values
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                T::from(x).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to cast X value at index {} to type T",
                        i
                    ))
                })
            })
            .collect::<StatsResult<Vec<T>>>()?;

        let y_cast: Vec<T> = y_values
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                T::from(y).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to cast Y value at index {} to type T",
                        i
                    ))
                })
            })
            .collect::<StatsResult<Vec<T>>>()?;

        // Calculate means
        let n_as_t = T::from(n).ok_or_else(|| {
            StatsError::conversion_error(format!("Failed to convert {} to type T", n))
        })?;
        let x_mean = x_cast.iter().fold(T::zero(), |acc, &x| acc + x) / n_as_t;
        let y_mean = y_cast.iter().fold(T::zero(), |acc, &y| acc + y) / n_as_t;

        // Calculate variance and covariance
        let mut sum_xy = T::zero();
        let mut sum_xx = T::zero();
        let mut sum_yy = T::zero();

        for i in 0..n {
            let x_diff = x_cast[i] - x_mean;
            let y_diff = y_cast[i] - y_mean;

            sum_xy = sum_xy + (x_diff * y_diff);
            sum_xx = sum_xx + (x_diff * x_diff);
            sum_yy = sum_yy + (y_diff * y_diff);
        }

        // Check if there's any variance in x
        if sum_xx == T::zero() {
            return Err(StatsError::invalid_parameter(
                "No variance in X values, cannot fit regression line",
            ));
        }

        // Calculate slope and intercept
        self.slope = sum_xy / sum_xx;
        self.intercept = y_mean - (self.slope * x_mean);
        self.x_mean = x_mean;
        self.sum_xx = sum_xx;

        // Calculate R²
        self.r_squared = (sum_xy * sum_xy) / (sum_xx * sum_yy);

        // Calculate residuals and standard error
        let mut sum_squared_residuals = T::zero();
        for i in 0..n {
            let predicted = self.predict_t(x_cast[i]);
            let residual = y_cast[i] - predicted;
            sum_squared_residuals = sum_squared_residuals + (residual * residual);
        }

        // Standard error of the estimate
        if n > 2 {
            let two = T::from(2)
                .ok_or_else(|| StatsError::conversion_error("Failed to convert 2 to type T"))?;
            let n_minus_two = n_as_t - two;
            self.standard_error = (sum_squared_residuals / n_minus_two).sqrt();
        } else {
            self.standard_error = T::zero();
        }

        Ok(())
    }

    /// Predict y value for a given x using the fitted model (internal version with type T)
    fn predict_t(&self, x: T) -> T {
        self.intercept + (self.slope * x)
    }

    /// Predict y value for a given x using the fitted model
    ///
    /// # Arguments
    /// * `x` - The x value to predict for
    ///
    /// # Returns
    /// * `StatsResult<T>` - The predicted y value
    ///
    /// # Errors
    /// Returns `StatsError::NotFitted` if the model has not been fitted (n == 0).
    /// Returns `StatsError::ConversionError` if type conversion fails.
    ///
    /// # Examples
    /// ```
    /// use rs_stats::regression::linear_regression::LinearRegression;
    ///
    /// let mut model = LinearRegression::<f64>::new();
    /// model.fit(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
    ///
    /// let prediction = model.predict(4.0).unwrap();
    /// assert!((prediction - 8.0).abs() < 1e-10);
    /// ```
    pub fn predict<U>(&self, x: U) -> StatsResult<T>
    where
        U: NumCast + Copy,
    {
        if self.n == 0 {
            return Err(StatsError::not_fitted(
                "Model has not been fitted. Call fit() before predicting.",
            ));
        }

        let x_cast: T = T::from(x)
            .ok_or_else(|| StatsError::conversion_error("Failed to convert x value to type T"))?;

        Ok(self.predict_t(x_cast))
    }

    /// Calculate predictions for multiple x values
    ///
    /// # Arguments
    /// * `x_values` - Slice of x values to predict for
    ///
    /// # Returns
    /// * `StatsResult<Vec<T>>` - Vector of predicted y values
    ///
    /// # Errors
    /// Returns `StatsError::NotFitted` if the model has not been fitted.
    /// Returns `StatsError::ConversionError` if type conversion fails for any value.
    ///
    /// # Examples
    /// ```
    /// use rs_stats::regression::linear_regression::LinearRegression;
    ///
    /// let mut model = LinearRegression::<f64>::new();
    /// model.fit(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
    ///
    /// let predictions = model.predict_many(&[4.0, 5.0]).unwrap();
    /// assert_eq!(predictions.len(), 2);
    /// ```
    pub fn predict_many<U>(&self, x_values: &[U]) -> StatsResult<Vec<T>>
    where
        U: NumCast + Copy + Send + Sync,
        T: Send + Sync,
    {
        // Each prediction is two flops; below this size the rayon dispatch
        // overhead dwarfs the work itself.
        #[cfg(feature = "parallel")]
        {
            const PAR_THRESHOLD: usize = 10_000;
            if x_values.len() >= PAR_THRESHOLD {
                return x_values.par_iter().map(|&x| self.predict(x)).collect();
            }
        }
        x_values.iter().map(|&x| self.predict(x)).collect()
    }

    /// Shared machinery for confidence / prediction intervals.
    ///
    /// `extra` is 0 for the CI of the mean response and 1 for a prediction
    /// interval (adds the variance of a single new observation).
    fn t_interval<U>(&self, x: U, confidence_level: f64, extra: f64) -> StatsResult<(T, T)>
    where
        U: NumCast + Copy,
    {
        if self.n < 3 {
            return Err(StatsError::invalid_input(
                "Need at least 3 data points to calculate confidence interval",
            ));
        }
        if !(0.0..1.0).contains(&confidence_level) || confidence_level <= 0.0 {
            return Err(StatsError::invalid_parameter(format!(
                "Confidence level must be in (0, 1), got {confidence_level}"
            )));
        }
        if self.sum_xx <= T::zero() {
            return Err(StatsError::invalid_input(
                "Model was fitted without X dispersion info (pre-v3.1 saved model?); refit before computing intervals",
            ));
        }

        let x_cast: T = T::from(x)
            .ok_or_else(|| StatsError::conversion_error("Failed to convert x value to type T"))?;

        // Student-t critical value with n − 2 degrees of freedom. The old
        // implementation used hardcoded *normal* z-scores, which understate
        // the interval for small n — exactly where intervals matter most.
        use crate::distributions::student_t::StudentT;
        use crate::distributions::traits::Distribution as _;
        let df = (self.n - 2) as f64;
        let t_crit = StudentT::new(0.0, 1.0, df)?.inverse_cdf(0.5 * (1.0 + confidence_level))?;
        let t_crit: T = T::from(t_crit).ok_or_else(|| {
            StatsError::conversion_error("Failed to convert t quantile to type T")
        })?;

        // SE of the estimate at x: s·√(extra + 1/n + (x−x̄)²/Sxx).
        // The band widens away from x̄ — a constant ±t·s is neither a CI of
        // the mean response nor a prediction interval.
        let n_t = T::from(self.n)
            .ok_or_else(|| StatsError::conversion_error("Failed to convert n to type T"))?;
        let extra_t = T::from(extra)
            .ok_or_else(|| StatsError::conversion_error("Failed to convert constant to type T"))?;
        let dx = x_cast - self.x_mean;
        let se_at_x =
            self.standard_error * (extra_t + T::one() / n_t + dx * dx / self.sum_xx).sqrt();

        let predicted = self.predict_t(x_cast);
        let margin = t_crit * se_at_x;
        Ok((predicted - margin, predicted + margin))
    }

    /// Confidence interval for the **mean response** `E[Y | x]` at `x`.
    ///
    /// Uses Student-t quantiles with `n − 2` degrees of freedom and the
    /// standard error `s·√(1/n + (x−x̄)²/Sxx)`, so the band widens away
    /// from the centre of the data. Any `confidence_level` in (0, 1) is
    /// accepted (0.95 for 95% confidence).
    ///
    /// For bounds on a single future *observation*, use
    /// [`prediction_interval`](Self::prediction_interval) instead.
    ///
    /// # Returns
    /// * `StatsResult<(T, T)>` - Tuple of (lower_bound, upper_bound), or an error if invalid
    ///
    /// # Errors
    /// Returns `StatsError::InvalidInput` if there are fewer than 3 data points.
    /// Returns `StatsError::InvalidParameter` if `confidence_level` is not in (0, 1).
    /// Returns `StatsError::ConversionError` if value conversion fails.
    pub fn confidence_interval<U>(&self, x: U, confidence_level: f64) -> StatsResult<(T, T)>
    where
        U: NumCast + Copy,
    {
        self.t_interval(x, confidence_level, 0.0)
    }

    /// Prediction interval for a **single future observation** at `x`.
    ///
    /// Wider than [`confidence_interval`](Self::confidence_interval): adds
    /// the variance of one new observation, `s·√(1 + 1/n + (x−x̄)²/Sxx)`.
    ///
    /// # Errors
    /// Same as [`confidence_interval`](Self::confidence_interval).
    pub fn prediction_interval<U>(&self, x: U, confidence_level: f64) -> StatsResult<(T, T)>
    where
        U: NumCast + Copy,
    {
        self.t_interval(x, confidence_level, 1.0)
    }

    /// Get the correlation coefficient (r)
    ///
    /// The correlation coefficient ranges from -1 to 1, indicating the strength
    /// and direction of the linear relationship between x and y.
    ///
    /// # Returns
    /// * `StatsResult<T>` - The correlation coefficient
    ///
    /// # Errors
    /// Returns `StatsError::NotFitted` if the model has not been fitted (n == 0).
    ///
    /// # Examples
    /// ```
    /// use rs_stats::regression::linear_regression::LinearRegression;
    ///
    /// let mut model = LinearRegression::<f64>::new();
    /// model.fit(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
    ///
    /// let r = model.correlation_coefficient().unwrap();
    /// assert!((r - 1.0).abs() < 1e-10); // Perfect positive correlation
    /// ```
    pub fn correlation_coefficient(&self) -> StatsResult<T> {
        if self.n == 0 {
            return Err(StatsError::not_fitted(
                "Model has not been fitted. Call fit() before getting correlation coefficient.",
            ));
        }
        let r = self.r_squared.sqrt();
        Ok(if self.slope >= T::zero() { r } else { -r })
    }
}

/// Model persistence — requires the `serde` feature.
#[cfg(feature = "serde")]
impl<T> LinearRegression<T>
where
    T: Float + Debug + Default + NumCast + Serialize + for<'de> Deserialize<'de>,
{
    /// Save the model to a file
    ///
    /// # Arguments
    /// * `path` - Path where to save the model
    ///
    /// # Returns
    /// * `Result<(), io::Error>` - Ok if successful, Err with IO error if saving fails
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let file = File::create(path)?;
        // Use JSON format for human-readability
        serde_json::to_writer(file, self).map_err(io::Error::other)
    }

    /// Save the model in binary format
    ///
    /// # Arguments
    /// * `path` - Path where to save the model
    ///
    /// # Returns
    /// * `Result<(), io::Error>` - Ok if successful, Err with IO error if saving fails
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let file = File::create(path)?;
        // Use bincode for more compact binary format
        bincode::serialize_into(file, self).map_err(io::Error::other)
    }

    /// Load a model from a file
    ///
    /// # Arguments
    /// * `path` - Path to the saved model file
    ///
    /// # Returns
    /// * `Result<Self, io::Error>` - Loaded model or IO error
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        // Try to load as JSON format
        serde_json::from_reader(file).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Load a model from a binary file
    ///
    /// # Arguments
    /// * `path` - Path to the saved model file
    ///
    /// # Returns
    /// * `Result<Self, io::Error>` - Loaded model or IO error
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        // Try to load as bincode format
        bincode::deserialize_from(file).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Save the model to a string in JSON format
    ///
    /// # Returns
    /// * `Result<String, String>` - JSON string representation or error message
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("Failed to serialize model: {}", e))
    }

    /// Load a model from a JSON string
    ///
    /// # Arguments
    /// * `json` - JSON string containing the model data
    ///
    /// # Returns
    /// * `Result<Self, String>` - Loaded model or error message
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to deserialize model: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_equal;
    #[cfg(feature = "serde")]
    use tempfile::tempdir;

    #[test]
    fn test_simple_regression_f64() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(approx_equal(model.slope, 2.0, Some(1e-6)));
        assert!(approx_equal(model.intercept, 0.0, Some(1e-6)));
        assert!(approx_equal(model.r_squared, 1.0, Some(1e-6)));
    }

    #[test]
    fn test_simple_regression_f32() {
        let x = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
        let y = vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32];

        let mut model = LinearRegression::<f32>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(approx_equal(model.slope, 2.0f32, Some(1e-6)));
        assert!(approx_equal(model.intercept, 0.0f32, Some(1e-6)));
        assert!(approx_equal(model.r_squared, 1.0f32, Some(1e-6)));
    }

    #[test]
    fn test_integer_data() {
        let x = vec![1, 2, 3, 4, 5];
        let y = vec![2, 4, 6, 8, 10];

        let mut model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(approx_equal(model.slope, 2.0, Some(1e-6)));
        assert!(approx_equal(model.intercept, 0.0, Some(1e-6)));
        assert!(approx_equal(model.r_squared, 1.0, Some(1e-6)));
    }

    #[test]
    fn test_mixed_types() {
        let x = vec![1u32, 2u32, 3u32, 4u32, 5u32];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];

        let mut model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(model.slope > 1.9 && model.slope < 2.1);
        assert!(model.intercept > -0.1 && model.intercept < 0.1);
        assert!(model.r_squared > 0.99);
    }

    #[test]
    fn test_prediction() {
        let x = vec![1, 2, 3, 4, 5];
        let y = vec![2, 4, 6, 8, 10];

        let mut model = LinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        assert!(approx_equal(model.predict(6u32).unwrap(), 12.0, Some(1e-6)));
        assert!(approx_equal(model.predict(0i32).unwrap(), 0.0, Some(1e-6)));
    }

    #[test]
    fn test_invalid_inputs() {
        let x = vec![1, 2, 3];
        let y = vec![2, 4];

        let mut model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_constant_x() {
        let x = vec![1, 1, 1];
        let y = vec![2, 3, 4];

        let mut model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_save_load_json() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.json");

        // Create and fit a model
        let mut model = LinearRegression::<f64>::new();
        model
            .fit(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0])
            .unwrap();

        // Save the model
        let save_result = model.save(&file_path);
        assert!(save_result.is_ok());

        // Load the model
        let loaded_model = LinearRegression::<f64>::load(&file_path);
        assert!(loaded_model.is_ok());
        let loaded = loaded_model.unwrap();

        // Check that the loaded model has the same parameters
        assert!(approx_equal(loaded.slope, model.slope, Some(1e-6)));
        assert!(approx_equal(loaded.intercept, model.intercept, Some(1e-6)));
        assert!(approx_equal(loaded.r_squared, model.r_squared, Some(1e-6)));
        assert_eq!(loaded.n, model.n);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_save_load_binary() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.bin");

        // Create and fit a model
        let mut model = LinearRegression::<f64>::new();
        model
            .fit(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0])
            .unwrap();

        // Save the model
        let save_result = model.save_binary(&file_path);
        assert!(save_result.is_ok());

        // Load the model
        let loaded_model = LinearRegression::<f64>::load_binary(&file_path);
        assert!(loaded_model.is_ok());
        let loaded = loaded_model.unwrap();

        // Check that the loaded model has the same parameters
        assert!(approx_equal(loaded.slope, model.slope, Some(1e-6)));
        assert!(approx_equal(loaded.intercept, model.intercept, Some(1e-6)));
        assert!(approx_equal(loaded.r_squared, model.r_squared, Some(1e-6)));
        assert_eq!(loaded.n, model.n);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_json_serialization() {
        // Create and fit a model
        let mut model = LinearRegression::<f64>::new();
        model
            .fit(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0])
            .unwrap();

        // Serialize to JSON string
        let json_result = model.to_json();
        assert!(json_result.is_ok());
        let json_str = json_result.unwrap();

        // Deserialize from JSON string
        let loaded_model = LinearRegression::<f64>::from_json(&json_str);
        assert!(loaded_model.is_ok());
        let loaded = loaded_model.unwrap();

        // Check that the loaded model has the same parameters
        assert!(approx_equal(loaded.slope, model.slope, Some(1e-6)));
        assert!(approx_equal(loaded.intercept, model.intercept, Some(1e-6)));
        assert!(approx_equal(loaded.r_squared, model.r_squared, Some(1e-6)));
        assert_eq!(loaded.n, model.n);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_load_nonexistent_file() {
        // Test loading from a file that doesn't exist
        let result = LinearRegression::<f64>::load("/nonexistent/path/model.json");
        assert!(result.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_load_binary_nonexistent_file() {
        // Test loading from a binary file that doesn't exist
        let result = LinearRegression::<f64>::load_binary("/nonexistent/path/model.bin");
        assert!(result.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_from_json_invalid_json() {
        // Test deserializing from invalid JSON
        let invalid_json = "{invalid json}";
        let result = LinearRegression::<f64>::from_json(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_when_not_fitted() {
        // Test that predict returns an error when model is not fitted
        let model = LinearRegression::<f64>::new();
        let result = model.predict(5.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::NotFitted { .. }));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_save_invalid_path() {
        // Test saving to an invalid path (non-existent directory)
        let mut model = LinearRegression::<f64>::new();
        model.fit(&[1.0, 2.0], &[2.0, 4.0]).unwrap();

        let invalid_path = std::path::Path::new("/nonexistent/directory/model.json");
        let result = model.save(invalid_path);
        assert!(
            result.is_err(),
            "Saving to invalid path should return error"
        );
    }

    #[test]
    fn test_fit_standard_error_n_less_than_or_equal_two() {
        // Test the branch where n <= 2 (standard_error = 0)
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0];
        let y = vec![2.0, 4.0];
        model.fit(&x, &y).unwrap();

        // When n = 2, standard_error should be 0
        assert_eq!(model.standard_error, 0.0);
    }

    #[test]
    fn test_fit_standard_error_n_greater_than_two() {
        // Test the branch where n > 2 (standard_error calculated)
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        model.fit(&x, &y).unwrap();

        // When n > 2, standard_error should be calculated
        assert!(model.standard_error >= 0.0);
    }

    #[test]
    fn test_confidence_interval_n_less_than_three() {
        // Test confidence_interval with n < 3
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0];
        let y = vec![2.0, 4.0];
        model.fit(&x, &y).unwrap();

        let result = model.confidence_interval(3.0, 0.95);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_confidence_interval_levels() {
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.1, 3.9, 6.2, 7.8];
        model.fit(&x, &y).unwrap();

        // Any level in (0, 1) is accepted — including non-standard ones.
        let (lo85, hi85) = model.confidence_interval(3.0, 0.85).unwrap();
        let (lo95, hi95) = model.confidence_interval(3.0, 0.95).unwrap();
        assert!(lo95 < lo85 && hi85 < hi95, "higher level ⇒ wider interval");

        // Levels outside (0, 1) are rejected.
        for bad in [0.0, 1.0, 1.5, -0.1] {
            assert!(matches!(
                model.confidence_interval(3.0, bad).unwrap_err(),
                StatsError::InvalidParameter { .. }
            ));
        }
    }

    #[test]
    fn test_interval_geometry() {
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 12.2, 13.8];
        model.fit(&x, &y).unwrap();

        // The CI band must widen away from x̄ (= 4.0 here).
        let w = |x0: f64| {
            let (lo, hi) = model.confidence_interval(x0, 0.95).unwrap();
            hi - lo
        };
        assert!(w(1.0) > w(4.0) && w(7.0) > w(4.0));

        // A prediction interval is strictly wider than the CI of the mean.
        let (ci_lo, ci_hi) = model.confidence_interval(4.0, 0.95).unwrap();
        let (pi_lo, pi_hi) = model.prediction_interval(4.0, 0.95).unwrap();
        assert!(pi_lo < ci_lo && ci_hi < pi_hi);
    }

    #[test]
    fn test_confidence_interval_supported_levels() {
        // Test all supported confidence levels
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        model.fit(&x, &y).unwrap();

        for level in [0.90, 0.95, 0.99] {
            let result = model.confidence_interval(3.0, level);
            assert!(
                result.is_ok(),
                "Confidence level {} should be supported",
                level
            );
            let (lower, upper) = result.unwrap();
            assert!(lower <= upper, "Lower bound should be <= upper bound");
        }
    }

    #[test]
    fn test_correlation_coefficient_positive_slope() {
        // Test correlation_coefficient with positive slope
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        model.fit(&x, &y).unwrap();

        let r = model.correlation_coefficient().unwrap();
        assert!(
            r >= 0.0,
            "Correlation should be positive for positive slope"
        );
    }

    #[test]
    fn test_correlation_coefficient_negative_slope() {
        // Test correlation_coefficient with negative slope
        let mut model = LinearRegression::<f64>::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![6.0, 4.0, 2.0];
        model.fit(&x, &y).unwrap();

        let r = model.correlation_coefficient().unwrap();
        assert!(
            r <= 0.0,
            "Correlation should be negative for negative slope"
        );
    }

    #[test]
    fn test_correlation_coefficient_not_fitted() {
        // Test correlation_coefficient when model is not fitted
        let model = LinearRegression::<f64>::new();
        let result = model.correlation_coefficient();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::NotFitted { .. }));
    }

    #[test]
    fn test_predict_many_not_fitted() {
        // Test predict_many when model is not fitted
        let model = LinearRegression::<f64>::new();
        let result = model.predict_many(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::NotFitted { .. }));
    }

    #[test]
    fn test_predict_many_success() {
        // Test predict_many with valid data
        let mut model = LinearRegression::<f64>::new();
        model.fit(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();

        let predictions = model.predict_many(&[4.0, 5.0]).unwrap();
        assert_eq!(predictions.len(), 2);
        assert!((predictions[0] - 8.0).abs() < 1e-10);
        assert!((predictions[1] - 10.0).abs() < 1e-10);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_load_invalid_json() {
        // Test loading invalid JSON
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid.json");

        // Write invalid JSON
        std::fs::write(&file_path, "invalid json content").unwrap();

        let result = LinearRegression::<f64>::load(&file_path);
        assert!(result.is_err(), "Loading invalid JSON should return error");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_from_json_invalid() {
        // Test deserializing invalid JSON string
        let invalid_json = "not valid json";
        let result = LinearRegression::<f64>::from_json(invalid_json);
        assert!(
            result.is_err(),
            "Deserializing invalid JSON should return error"
        );
    }
}
