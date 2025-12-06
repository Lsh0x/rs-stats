// src/regression/multiple_linear_regression.rs

use crate::error::{StatsError, StatsResult};
use num_traits::{Float, NumCast};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::io::{self};
use std::path::Path;

/// Multiple linear regression model that fits a hyperplane to multivariate data points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleLinearRegression<T = f64>
where
    T: Float + Debug + Default + Serialize,
{
    /// Regression coefficients, including intercept as the first element
    pub coefficients: Vec<T>,
    /// Coefficient of determination (R²) - goodness of fit
    pub r_squared: T,
    /// Adjusted R² which accounts for the number of predictors
    pub adjusted_r_squared: T,
    /// Standard error of the estimate
    pub standard_error: T,
    /// Number of data points used for regression
    pub n: usize,
    /// Number of predictor variables (excluding intercept)
    pub p: usize,
}

impl<T> Default for MultipleLinearRegression<T>
where
    T: Float + Debug + Default + NumCast + Serialize + for<'de> Deserialize<'de>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MultipleLinearRegression<T>
where
    T: Float + Debug + Default + NumCast + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new multiple linear regression model without fitting any data
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            r_squared: T::zero(),
            adjusted_r_squared: T::zero(),
            standard_error: T::zero(),
            n: 0,
            p: 0,
        }
    }

    /// Fit a multiple linear regression model to the provided data
    ///
    /// # Arguments
    /// * `x_values` - 2D array where each row is an observation and each column is a predictor
    /// * `y_values` - Dependent variable values (observations)
    ///
    /// # Returns
    /// * `StatsResult<()>` - Ok if successful, Err with StatsError if the inputs are invalid
    ///
    /// # Errors
    /// Returns `StatsError::EmptyData` if input arrays are empty.
    /// Returns `StatsError::DimensionMismatch` if X and Y arrays have different lengths.
    /// Returns `StatsError::InvalidInput` if rows in X have inconsistent lengths.
    /// Returns `StatsError::ConversionError` if value conversion fails.
    /// Returns `StatsError::MathematicalError` if the linear system cannot be solved.
    pub fn fit<U, V>(&mut self, x_values: &[Vec<U>], y_values: &[V]) -> StatsResult<()>
    where
        U: NumCast + Copy,
        V: NumCast + Copy,
    {
        // Validate inputs
        if x_values.is_empty() || y_values.is_empty() {
            return Err(StatsError::empty_data(
                "Cannot fit regression with empty arrays"
            ));
        }

        if x_values.len() != y_values.len() {
            return Err(StatsError::dimension_mismatch(format!(
                "Number of observations in X and Y must match (got {} and {})",
                x_values.len(),
                y_values.len()
            )));
        }

        self.n = x_values.len();

        // Check that all rows in x_values have the same length
        if x_values.is_empty() {
            return Err(StatsError::empty_data("X values array is empty"));
        }

        self.p = x_values[0].len();

        for (i, row) in x_values.iter().enumerate() {
            if row.len() != self.p {
                return Err(StatsError::invalid_input(format!(
                    "All rows in X must have the same number of features (row {} has {} features, expected {})",
                    i, row.len(), self.p
                )));
            }
        }

        // Convert input arrays to T type
        let mut x_cast: Vec<Vec<T>> = Vec::with_capacity(self.n);
        for (row_idx, row) in x_values.iter().enumerate() {
            let row_cast: StatsResult<Vec<T>> = row
                .iter()
                .enumerate()
                .map(|(col_idx, &x)| {
                    T::from(x).ok_or_else(|| StatsError::conversion_error(format!(
                        "Failed to cast X value at row {}, column {} to type T",
                        row_idx, col_idx
                    )))
                })
                .collect();
            x_cast.push(row_cast?);
        }

        let y_cast: Vec<T> = y_values
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                T::from(y).ok_or_else(|| StatsError::conversion_error(format!(
                    "Failed to cast Y value at index {} to type T",
                    i
                )))
            })
            .collect::<StatsResult<Vec<T>>>()?;

        // Augment the X matrix with a column of 1s for the intercept
        let mut augmented_x = Vec::with_capacity(self.n);
        for row in &x_cast {
            let mut augmented_row = Vec::with_capacity(self.p + 1);
            augmented_row.push(T::one()); // Intercept term
            augmented_row.extend_from_slice(row);
            augmented_x.push(augmented_row);
        }

        // Compute X^T * X
        let xt_x = self.matrix_multiply_transpose(&augmented_x, &augmented_x);

        // Compute X^T * y
        let xt_y = self.vector_multiply_transpose(&augmented_x, &y_cast);

        // Solve the normal equations: (X^T * X) * β = X^T * y for β
        match self.solve_linear_system(&xt_x, &xt_y) {
            Ok(solution) => {
                self.coefficients = solution;
            }
            Err(e) => return Err(e),
        }

        // Calculate fitted values and R²
        let n_as_t = T::from(self.n).ok_or_else(|| StatsError::conversion_error(format!(
            "Failed to convert {} to type T",
            self.n
        )))?;
        let y_mean = y_cast.iter().fold(T::zero(), |acc, &y| acc + y) / n_as_t;

        let mut ss_total = T::zero();
        let mut ss_residual = T::zero();

        for i in 0..self.n {
            let predicted = self.predict_t(&x_cast[i]);
            let residual = y_cast[i] - predicted;

            ss_residual = ss_residual + (residual * residual);
            let diff = y_cast[i] - y_mean;
            ss_total = ss_total + (diff * diff);
        }

        // Calculate R² and adjusted R²
        if ss_total > T::zero() {
            self.r_squared = T::one() - (ss_residual / ss_total);

            // Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
            if self.n > self.p + 1 {
                let n_minus_1 = T::from(self.n - 1).ok_or_else(|| StatsError::conversion_error(format!(
                    "Failed to convert {} to type T",
                    self.n - 1
                )))?;
                let n_minus_p_minus_1 = T::from(self.n - self.p - 1).ok_or_else(|| StatsError::conversion_error(format!(
                    "Failed to convert {} to type T",
                    self.n - self.p - 1
                )))?;

                self.adjusted_r_squared =
                    T::one() - ((T::one() - self.r_squared) * n_minus_1 / n_minus_p_minus_1);
            }
        }

        // Calculate standard error
        if self.n > self.p + 1 {
            let n_minus_p_minus_1 = T::from(self.n - self.p - 1).ok_or_else(|| StatsError::conversion_error(format!(
                "Failed to convert {} to type T",
                self.n - self.p - 1
            )))?;
            self.standard_error = (ss_residual / n_minus_p_minus_1).sqrt();
        }

        Ok(())
    }

    /// Predict y value for a given set of x values using the fitted model (internal version with type T)
    fn predict_t(&self, x: &[T]) -> T {
        if x.len() != self.p || self.coefficients.is_empty() {
            return T::nan();
        }

        // First coefficient is the intercept
        let mut result = self.coefficients[0];

        // Add the weighted features
        for (i, &xi) in x.iter().enumerate().take(self.p) {
            result = result + (self.coefficients[i + 1] * xi);
        }

        result
    }

    /// Predict y value for a given set of x values using the fitted model
    ///
    /// # Arguments
    /// * `x` - Vector of x values for prediction
    ///
    /// # Returns
    /// * The predicted y value
    pub fn predict<U>(&self, x: &[U]) -> T
    where
        U: NumCast + Copy,
    {
        if x.len() != self.p {
            return T::nan();
        }

        // Convert input to T type
        let x_cast: Result<Vec<T>, ()> = x.iter().map(|&val| T::from(val).ok_or(())).collect();

        match x_cast {
            Ok(x_t) => self.predict_t(&x_t),
            Err(_) => T::nan(),
        }
    }

    /// Calculate predictions for multiple observations
    ///
    /// # Arguments
    /// * `x_values` - 2D array of feature values for prediction
    ///
    /// # Returns
    /// * Vector of predicted y values
    pub fn predict_many<U>(&self, x_values: &[Vec<U>]) -> Vec<T>
    where
        U: NumCast + Copy,
    {
        x_values.iter().map(|x| self.predict(x)).collect()
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
        serde_json::to_writer(file, self).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
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
        bincode::serialize_into(file, self).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
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

    // Helper function: Matrix multiplication where one matrix is transposed: A^T * B
    fn matrix_multiply_transpose(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        let a_rows = a.len();
        let a_cols = if a_rows > 0 { a[0].len() } else { 0 };
        let b_rows = b.len();
        let b_cols = if b_rows > 0 { b[0].len() } else { 0 };

        // Result will be a_cols × b_cols
        let mut result = vec![vec![T::zero(); b_cols]; a_cols];

        for (i, result_row) in result.iter_mut().enumerate().take(a_cols) {
            for (j, result_elem) in result_row.iter_mut().enumerate().take(b_cols) {
                let mut sum = T::zero();
                for k in 0..a_rows {
                    sum = sum + (a[k][i] * b[k][j]);
                }
                *result_elem = sum;
            }
        }

        result
    }

    // Helper function: Multiply transposed matrix by vector: A^T * y
    fn vector_multiply_transpose(&self, a: &[Vec<T>], y: &[T]) -> Vec<T> {
        let a_rows = a.len();
        let a_cols = if a_rows > 0 { a[0].len() } else { 0 };

        let mut result = vec![T::zero(); a_cols];

        for (i, result_item) in result.iter_mut().enumerate().take(a_cols) {
            let mut sum = T::zero();
            for j in 0..a_rows {
                sum = sum + (a[j][i] * y[j]);
            }
            *result_item = sum;
        }

        result
    }

    // Helper function: Solve a system of linear equations using Gaussian elimination
    fn solve_linear_system(&self, a: &[Vec<T>], b: &[T]) -> StatsResult<Vec<T>> {
        let n = a.len();
        if n == 0 || a[0].len() != n || b.len() != n {
            return Err(StatsError::dimension_mismatch(format!(
                "Invalid matrix dimensions for linear system solving: A is {}x{}, b has {} elements",
                n,
                if n > 0 { a[0].len() } else { 0 },
                b.len()
            )));
        }

        // Create augmented matrix [A|b]
        let mut aug = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = a[i].clone();
            row.push(b[i]);
            aug.push(row);
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = aug[i][i].abs();

            for (j, row) in aug.iter().enumerate().skip(i + 1).take(n - (i + 1)) {
                let abs_val = row[i].abs();
                if abs_val > max_val {
                    max_row = j;
                    max_val = abs_val;
                }
            }

            let epsilon: T = T::from(1e-10).ok_or_else(|| StatsError::conversion_error(
                "Failed to convert epsilon (1e-10) to type T"
            ))?;
            if max_val < epsilon {
                return Err(StatsError::mathematical_error(
                    "Matrix is singular or near-singular, cannot solve linear system"
                ));
            }

            // Swap rows if needed
            if max_row != i {
                aug.swap(i, max_row);
            }

            // Eliminate below
            for j in (i + 1)..n {
                let factor = aug[j][i] / aug[i][i];

                for k in i..(n + 1) {
                    aug[j][k] = aug[j][k] - (factor * aug[i][k]);
                }
            }
        }

        // Back substitution
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];

            for (j, &x_val) in x.iter().enumerate().skip(i + 1).take(n - (i + 1)) {
                sum = sum - (aug[i][j] * x_val);
            }

            x[i] = sum / aug[i][i];
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::numeric::approx_equal;
    use tempfile::tempdir;

    #[test]
    fn test_simple_multi_regression_f64() {
        // Simple case: y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ];
        let y = vec![9.0, 8.0, 16.0, 15.0];

        let mut model = MultipleLinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(model.coefficients.len() == 3);
        assert!(approx_equal(model.coefficients[0], 1.0, Some(1e-6))); // intercept
        assert!(approx_equal(model.coefficients[1], 2.0, Some(1e-6))); // x1 coefficient
        assert!(approx_equal(model.coefficients[2], 3.0, Some(1e-6))); // x2 coefficient
        assert!(approx_equal(model.r_squared, 1.0, Some(1e-6)));
    }

    #[test]
    fn test_simple_multi_regression_f32() {
        // Simple case: y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![1.0f32, 2.0f32],
            vec![2.0f32, 1.0f32],
            vec![3.0f32, 3.0f32],
            vec![4.0f32, 2.0f32],
        ];
        let y = vec![9.0f32, 8.0f32, 16.0f32, 15.0f32];

        let mut model = MultipleLinearRegression::<f32>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(model.coefficients.len() == 3);
        assert!(approx_equal(model.coefficients[0], 1.0f32, Some(1e-4))); // intercept
        assert!(approx_equal(model.coefficients[1], 2.0f32, Some(1e-4))); // x1 coefficient
        assert!(approx_equal(model.coefficients[2], 3.0f32, Some(1e-4))); // x2 coefficient
        assert!(approx_equal(model.r_squared, 1.0f32, Some(1e-4)));
    }

    #[test]
    fn test_integer_data() {
        // Simple case: y = 2*x1 + 3*x2 + 1
        let x = vec![
            vec![1u32, 2u32],
            vec![2u32, 1u32],
            vec![3u32, 3u32],
            vec![4u32, 2u32],
        ];
        let y = vec![9i32, 8i32, 16i32, 15i32];

        let mut model = MultipleLinearRegression::<f64>::new();
        let result = model.fit(&x, &y);

        assert!(result.is_ok());
        assert!(model.coefficients.len() == 3);
        assert!(approx_equal(model.coefficients[0], 1.0, Some(1e-6))); // intercept
        assert!(approx_equal(model.coefficients[1], 2.0, Some(1e-6))); // x1 coefficient
        assert!(approx_equal(model.coefficients[2], 3.0, Some(1e-6))); // x2 coefficient
        assert!(approx_equal(model.r_squared, 1.0, Some(1e-6)));
    }

    #[test]
    fn test_prediction() {
        // Simple case: y = 2*x1 + 3*x2 + 1
        let x = vec![vec![1, 2], vec![2, 1], vec![3, 3], vec![4, 2]];
        let y = vec![9, 8, 16, 15];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        // Test prediction: 1 + 2*5 + 3*4 = 1 + 10 + 12 = 23
        assert!(approx_equal(model.predict(&[5u32, 4u32]), 23.0, Some(1e-6)));
    }

    #[test]
    fn test_prediction_many() {
        let x = vec![vec![1, 2], vec![2, 1], vec![3, 3]];
        let y = vec![9, 8, 16];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        let new_x = vec![vec![1u32, 2u32], vec![5u32, 4u32]];

        let predictions = model.predict_many(&new_x);
        assert_eq!(predictions.len(), 2);
        assert!(approx_equal(predictions[0], 9.0, Some(1e-6)));
        assert!(approx_equal(predictions[1], 23.0, Some(1e-6)));
    }

    #[test]
    fn test_save_load_json() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.json");

        // Create and fit a model
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ];
        let y = vec![9.0, 8.0, 16.0, 15.0];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        // Save the model
        let save_result = model.save(&file_path);
        assert!(save_result.is_ok());

        // Load the model
        let loaded_model = MultipleLinearRegression::<f64>::load(&file_path);
        assert!(loaded_model.is_ok());
        let loaded = loaded_model.unwrap();

        // Check that the loaded model has the same parameters
        assert_eq!(loaded.coefficients.len(), model.coefficients.len());
        for i in 0..model.coefficients.len() {
            assert!(approx_equal(
                loaded.coefficients[i],
                model.coefficients[i],
                Some(1e-6)
            ));
        }
        assert!(approx_equal(loaded.r_squared, model.r_squared, Some(1e-6)));
        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.p, model.p);
    }

    #[test]
    fn test_save_load_binary() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("model.bin");

        // Create and fit a model
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ];
        let y = vec![9.0, 8.0, 16.0, 15.0];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        // Save the model
        let save_result = model.save_binary(&file_path);
        assert!(save_result.is_ok());

        // Load the model
        let loaded_model = MultipleLinearRegression::<f64>::load_binary(&file_path);
        assert!(loaded_model.is_ok());
        let loaded = loaded_model.unwrap();

        // Check that the loaded model has the same parameters
        assert_eq!(loaded.coefficients.len(), model.coefficients.len());
        for i in 0..model.coefficients.len() {
            assert!(approx_equal(
                loaded.coefficients[i],
                model.coefficients[i],
                Some(1e-6)
            ));
        }
        assert!(approx_equal(loaded.r_squared, model.r_squared, Some(1e-6)));
        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.p, model.p);
    }

    #[test]
    fn test_json_serialization() {
        // Create and fit a model
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ];
        let y = vec![9.0, 8.0, 16.0, 15.0];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        // Serialize to JSON string
        let json_result = model.to_json();
        assert!(json_result.is_ok());
        let json_str = json_result.unwrap();

        // Deserialize from JSON string
        let loaded_model = MultipleLinearRegression::<f64>::from_json(&json_str);
        assert!(loaded_model.is_ok());
        let loaded = loaded_model.unwrap();

        // Check that the loaded model has the same parameters
        assert_eq!(loaded.coefficients.len(), model.coefficients.len());
        for i in 0..model.coefficients.len() {
            assert!(approx_equal(
                loaded.coefficients[i],
                model.coefficients[i],
                Some(1e-6)
            ));
        }
        assert!(approx_equal(loaded.r_squared, model.r_squared, Some(1e-6)));
        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.p, model.p);
    }
}
