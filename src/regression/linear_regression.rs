// src/regression/linear_regression.rs

use crate::error::{StatsError, StatsResult};
use num_traits::{Float, NumCast};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::io::{self};
use std::path::Path;

/// Linear regression model that fits a line to data points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression<T = f64>
where
    T: Float + Debug + Default + Serialize,
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
}

impl<T> Default for LinearRegression<T>
where
    T: Float + Debug + Default + NumCast + Serialize + for<'de> Deserialize<'de>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LinearRegression<T>
where
    T: Float + Debug + Default + NumCast + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new linear regression model without fitting any data
    pub fn new() -> Self {
        Self {
            slope: T::zero(),
            intercept: T::zero(),
            r_squared: T::zero(),
            standard_error: T::zero(),
            n: 0,
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
    /// * The predicted y value
    pub fn predict<U>(&self, x: U) -> T
    where
        U: NumCast + Copy,
    {
        let x_cast: T = match T::from(x) {
            Some(val) => val,
            None => return T::nan(),
        };

        self.predict_t(x_cast)
    }

    /// Calculate predictions for multiple x values
    ///
    /// # Arguments
    /// * `x_values` - Slice of x values to predict for
    ///
    /// # Returns
    /// * Vector of predicted y values
    pub fn predict_many<U>(&self, x_values: &[U]) -> Vec<T>
    where
        U: NumCast + Copy,
    {
        x_values.iter().map(|&x| self.predict(x)).collect()
    }

    /// Calculate confidence intervals for the regression line
    ///
    /// # Arguments
    /// * `x` - The x value to calculate confidence interval for
    /// * `confidence_level` - Confidence level (0.95 for 95% confidence)
    ///
    /// # Returns
    /// * `StatsResult<(T, T)>` - Tuple of (lower_bound, upper_bound), or an error if invalid
    ///
    /// # Errors
    /// Returns `StatsError::InvalidInput` if there are fewer than 3 data points.
    /// Returns `StatsError::InvalidParameter` if confidence level is not supported (only 0.90, 0.95, 0.99).
    /// Returns `StatsError::ConversionError` if value conversion fails.
    pub fn confidence_interval<U>(&self, x: U, confidence_level: f64) -> StatsResult<(T, T)>
    where
        U: NumCast + Copy,
    {
        if self.n < 3 {
            return Err(StatsError::invalid_input(
                "Need at least 3 data points to calculate confidence interval",
            ));
        }

        let x_cast: T = T::from(x)
            .ok_or_else(|| StatsError::conversion_error("Failed to convert x value to type T"))?;

        // Get the t-critical value based on degrees of freedom and confidence level
        // For simplicity, we'll use a normal approximation with standard errors
        let z_score: T = match confidence_level {
            0.90 => T::from(1.645).ok_or_else(|| {
                StatsError::conversion_error("Failed to convert z-score 1.645 to type T")
            })?,
            0.95 => T::from(1.96).ok_or_else(|| {
                StatsError::conversion_error("Failed to convert z-score 1.96 to type T")
            })?,
            0.99 => T::from(2.576).ok_or_else(|| {
                StatsError::conversion_error("Failed to convert z-score 2.576 to type T")
            })?,
            _ => {
                return Err(StatsError::invalid_parameter(format!(
                    "Unsupported confidence level: {}. Supported values: 0.90, 0.95, 0.99",
                    confidence_level
                )));
            }
        };

        let predicted = self.predict_t(x_cast);
        let margin = z_score * self.standard_error;

        Ok((predicted - margin, predicted + margin))
    }

    /// Get the correlation coefficient (r)
    pub fn correlation_coefficient(&self) -> T {
        let r = self.r_squared.sqrt();
        if self.slope >= T::zero() { r } else { -r }
    }

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
        serde_json::to_writer(file, self).map_err(|e| io::Error::other(e))
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
        bincode::serialize_into(file, self).map_err(|e| io::Error::other(e))
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
    use crate::utils::numeric::approx_equal;
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

        assert!(approx_equal(model.predict(6u32), 12.0, Some(1e-6)));
        assert!(approx_equal(model.predict(0i32), 0.0, Some(1e-6)));
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
}
