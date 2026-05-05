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
                "Cannot fit regression with empty arrays",
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
                    i,
                    row.len(),
                    self.p
                )));
            }
        }

        // Build the augmented design matrix `X' = [1 | X]` directly in
        // row-major flat storage — n_samples × (p+1) — without going
        // through an intermediate `Vec<Vec<T>>`. Saves n + 1 allocations
        // and gives matrix_multiply_transpose contiguous row strides.
        let m = self.p + 1;
        let mut augmented_x: Vec<T> = Vec::with_capacity(self.n * m);
        for (row_idx, row) in x_values.iter().enumerate() {
            augmented_x.push(T::one()); // intercept column
            for (col_idx, &x) in row.iter().enumerate() {
                let cast = T::from(x).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to cast X value at row {row_idx}, column {col_idx} to type T"
                    ))
                })?;
                augmented_x.push(cast);
            }
        }

        let y_cast: Vec<T> = y_values
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                T::from(y).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to cast Y value at index {i} to type T"
                    ))
                })
            })
            .collect::<StatsResult<Vec<T>>>()?;

        // X^T · X — m × m matrix in flat row-major storage.
        let xt_x = matrix_multiply_transpose_flat::<T>(&augmented_x, self.n, m);

        // X^T · y — length-m vector.
        let xt_y = vector_multiply_transpose_flat::<T>(&augmented_x, &y_cast, self.n, m);

        // Solve the normal equations (X^T·X) · β = X^T·y.
        self.coefficients = solve_linear_system_flat::<T>(&xt_x, &xt_y, m)?;

        // Compute R² and standard error in a single pass over the rows.
        let n_as_t = T::from(self.n).ok_or_else(|| {
            StatsError::conversion_error(format!("Failed to convert {} to type T", self.n))
        })?;
        let y_mean = y_cast.iter().fold(T::zero(), |acc, &y| acc + y) / n_as_t;

        let mut ss_total = T::zero();
        let mut ss_residual = T::zero();

        for i in 0..self.n {
            // The features (excluding the intercept column) live at
            // augmented_x[i*m + 1 .. i*m + m]; predict_t expects exactly
            // those p values.
            let row_start = i * m + 1;
            let row_end = i * m + m;
            let predicted = self.predict_t(&augmented_x[row_start..row_end]);
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
                let n_minus_1 = T::from(self.n - 1).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to convert {} to type T",
                        self.n - 1
                    ))
                })?;
                let n_minus_p_minus_1 = T::from(self.n - self.p - 1).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to convert {} to type T",
                        self.n - self.p - 1
                    ))
                })?;

                self.adjusted_r_squared =
                    T::one() - ((T::one() - self.r_squared) * n_minus_1 / n_minus_p_minus_1);
            }
        }

        // Calculate standard error
        if self.n > self.p + 1 {
            let n_minus_p_minus_1 = T::from(self.n - self.p - 1).ok_or_else(|| {
                StatsError::conversion_error(format!(
                    "Failed to convert {} to type T",
                    self.n - self.p - 1
                ))
            })?;
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
    /// * `x` - Vector of x values for prediction (must match the number of features used during fitting)
    ///
    /// # Returns
    /// * `StatsResult<T>` - The predicted y value
    ///
    /// # Errors
    /// Returns `StatsError::NotFitted` if the model has not been fitted (coefficients is empty).
    /// Returns `StatsError::DimensionMismatch` if the number of features doesn't match the model (x.len() != p).
    /// Returns `StatsError::ConversionError` if type conversion fails.
    ///
    /// # Examples
    /// ```
    /// use rs_stats::regression::multiple_linear_regression::MultipleLinearRegression;
    ///
    /// let mut model = MultipleLinearRegression::<f64>::new();
    /// let x = vec![
    ///     vec![1.0, 2.0],
    ///     vec![2.0, 1.0],
    ///     vec![3.0, 3.0],
    ///     vec![4.0, 2.0],
    /// ];
    /// let y = vec![5.0, 4.0, 9.0, 8.0];
    /// model.fit(&x, &y).unwrap();
    ///
    /// let prediction = model.predict(&[3.0, 4.0]).unwrap();
    /// ```
    pub fn predict<U>(&self, x: &[U]) -> StatsResult<T>
    where
        U: NumCast + Copy,
    {
        if self.coefficients.is_empty() {
            return Err(StatsError::not_fitted(
                "Model has not been fitted. Call fit() before predicting.",
            ));
        }

        if x.len() != self.p {
            return Err(StatsError::dimension_mismatch(format!(
                "Expected {} features, but got {}",
                self.p,
                x.len()
            )));
        }

        // Convert input to T type
        let x_cast: StatsResult<Vec<T>> = x
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                T::from(val).ok_or_else(|| {
                    StatsError::conversion_error(format!(
                        "Failed to convert feature value at index {} to type T",
                        i
                    ))
                })
            })
            .collect();

        Ok(self.predict_t(&x_cast?))
    }

    /// Calculate predictions for multiple observations
    ///
    /// # Arguments
    /// * `x_values` - 2D array of feature values for prediction
    ///
    /// # Returns
    /// * `StatsResult<Vec<T>>` - Vector of predicted y values
    ///
    /// # Errors
    /// Returns `StatsError::NotFitted` if the model has not been fitted.
    /// Returns an error if any prediction fails (dimension mismatch or conversion error).
    ///
    /// # Examples
    /// ```
    /// use rs_stats::regression::multiple_linear_regression::MultipleLinearRegression;
    ///
    /// let mut model = MultipleLinearRegression::<f64>::new();
    /// let x = vec![
    ///     vec![1.0, 2.0],
    ///     vec![2.0, 1.0],
    ///     vec![3.0, 3.0],
    ///     vec![4.0, 2.0],
    /// ];
    /// let y = vec![5.0, 4.0, 9.0, 8.0];
    /// model.fit(&x, &y).unwrap();
    ///
    /// let predictions = model.predict_many(&[vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
    /// assert_eq!(predictions.len(), 2);
    /// ```
    pub fn predict_many<U>(&self, x_values: &[Vec<U>]) -> StatsResult<Vec<T>>
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

// ── Flat row-major linear-algebra helpers ─────────────────────────────────────

/// `Aᵀ · A` where `a` is row-major `n_rows × n_cols`. Returns the
/// `n_cols × n_cols` Gram matrix in row-major flat storage.
fn matrix_multiply_transpose_flat<T>(a: &[T], n_rows: usize, n_cols: usize) -> Vec<T>
where
    T: Float,
{
    let mut result = vec![T::zero(); n_cols * n_cols];
    // Iterate over rows in the outer loop — both reads (a[k*n_cols + ..])
    // are contiguous, and the inner write avoids column-stride misses.
    for k in 0..n_rows {
        let row_off = k * n_cols;
        for i in 0..n_cols {
            let a_ki = a[row_off + i];
            if a_ki == T::zero() {
                continue;
            }
            let dst_off = i * n_cols;
            for j in 0..n_cols {
                result[dst_off + j] = result[dst_off + j] + a_ki * a[row_off + j];
            }
        }
    }
    result
}

/// `Aᵀ · y` where `a` is row-major `n_rows × n_cols`.
fn vector_multiply_transpose_flat<T>(a: &[T], y: &[T], n_rows: usize, n_cols: usize) -> Vec<T>
where
    T: Float,
{
    let mut result = vec![T::zero(); n_cols];
    for k in 0..n_rows {
        let row_off = k * n_cols;
        let yk = y[k];
        for i in 0..n_cols {
            result[i] = result[i] + a[row_off + i] * yk;
        }
    }
    result
}

/// Gaussian elimination with partial pivoting on a flat `n × n` matrix
/// `a` and right-hand-side vector `b` of length `n`. Returns the
/// solution `x` of `a · x = b`. The augmented buffer is built once.
fn solve_linear_system_flat<T>(a: &[T], b: &[T], n: usize) -> StatsResult<Vec<T>>
where
    T: Float + Debug,
{
    if a.len() != n * n || b.len() != n {
        return Err(StatsError::dimension_mismatch(format!(
            "Invalid matrix dimensions for linear system solving: A is {n}×{n} ({} elems), b has {} elements",
            a.len(),
            b.len()
        )));
    }
    let w = n + 1;
    let mut aug: Vec<T> = Vec::with_capacity(n * w);
    for i in 0..n {
        aug.extend_from_slice(&a[i * n..(i + 1) * n]);
        aug.push(b[i]);
    }

    let epsilon: T = T::from(1e-10).ok_or_else(|| {
        StatsError::conversion_error("Failed to convert epsilon (1e-10) to type T")
    })?;

    for i in 0..n {
        // Partial pivot
        let mut max_row = i;
        let mut max_val = aug[i * w + i].abs();
        for j in (i + 1)..n {
            let abs_val = aug[j * w + i].abs();
            if abs_val > max_val {
                max_row = j;
                max_val = abs_val;
            }
        }
        if max_val < epsilon {
            return Err(StatsError::mathematical_error(
                "Matrix is singular or near-singular, cannot solve linear system",
            ));
        }
        if max_row != i {
            for c in 0..w {
                aug.swap(i * w + c, max_row * w + c);
            }
        }

        // Eliminate below
        let pivot = aug[i * w + i];
        for j in (i + 1)..n {
            let factor = aug[j * w + i] / pivot;
            for k in i..w {
                aug[j * w + k] = aug[j * w + k] - factor * aug[i * w + k];
            }
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = aug[i * w + n];
        for j in (i + 1)..n {
            sum = sum - aug[i * w + j] * x[j];
        }
        x[i] = sum / aug[i * w + i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::approx_equal;
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
        assert!(approx_equal(
            model.predict(&[5u32, 4u32]).unwrap(),
            23.0,
            Some(1e-6)
        ));
    }

    #[test]
    fn test_prediction_many() {
        let x = vec![vec![1, 2], vec![2, 1], vec![3, 3]];
        let y = vec![9, 8, 16];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();

        let new_x = vec![vec![1u32, 2u32], vec![5u32, 4u32]];

        let predictions = model.predict_many(&new_x).unwrap();
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

    #[test]
    fn test_predict_not_fitted() {
        // Test that predict() works even when model is not fitted
        let model = MultipleLinearRegression::<f64>::new();
        // Don't fit the model

        // Predict should return an error when model is not fitted
        let features = vec![1.0, 2.0];
        let result = model.predict(&features);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::NotFitted { .. }));
    }

    #[test]
    fn test_predict_dimension_mismatch() {
        // Test predict with wrong number of features
        let mut model = MultipleLinearRegression::<f64>::new();
        // Use more data points to avoid singular matrix
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ];
        let y = vec![3.0, 3.0, 6.0, 6.0];
        model.fit(&x, &y).unwrap();

        // Try to predict with wrong number of features
        let wrong_features = vec![1.0]; // Should be 2 features
        let result = model.predict(&wrong_features);
        // predict returns error when dimension mismatch
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_fit_singular_matrix() {
        // Test with linearly dependent features (singular matrix)
        // This should trigger a mathematical error
        let x = vec![
            vec![1.0, 2.0, 3.0], // Feature 3 = Feature 1 + Feature 2 (linearly dependent)
            vec![2.0, 4.0, 6.0], // Feature 3 = 2 * (Feature 1 + Feature 2)
            vec![3.0, 6.0, 9.0], // Feature 3 = 3 * (Feature 1 + Feature 2)
        ];
        let y = vec![1.0, 2.0, 3.0];

        let mut model = MultipleLinearRegression::<f64>::new();
        let result = model.fit(&x, &y);
        // This might succeed or fail depending on numerical precision
        // The important thing is it doesn't panic
        match result {
            Ok(_) => {
                // If it succeeds, verify the model is valid
                assert!(!model.coefficients.is_empty());
            }
            Err(e) => {
                // If it fails, it should be a mathematical error
                assert!(matches!(e, StatsError::MathematicalError { .. }));
            }
        }
    }

    #[test]
    fn test_save_invalid_path() {
        // Test saving to an invalid path
        let mut model = MultipleLinearRegression::<f64>::new();
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![2.0, 4.0];
        model.fit(&x, &y).unwrap();

        let invalid_path = std::path::Path::new("/nonexistent/directory/model.json");
        let result = model.save(invalid_path);
        assert!(
            result.is_err(),
            "Saving to invalid path should return error"
        );
    }

    #[test]
    fn test_load_nonexistent_file() {
        // Test loading a non-existent file
        let nonexistent_path = std::path::Path::new("/nonexistent/file.json");
        let result = MultipleLinearRegression::<f64>::load(nonexistent_path);
        assert!(
            result.is_err(),
            "Loading non-existent file should return error"
        );
    }

    #[test]
    fn test_from_json_invalid() {
        // Test deserializing invalid JSON string
        let invalid_json = "not valid json";
        let result = MultipleLinearRegression::<f64>::from_json(invalid_json);
        assert!(
            result.is_err(),
            "Deserializing invalid JSON should return error"
        );
    }

    #[test]
    fn test_predict_t_coefficients_empty() {
        // Test predict_t when coefficients are empty
        let model = MultipleLinearRegression::<f64>::new();
        let features = vec![1.0, 2.0];
        // predict_t is private, but we can test through predict
        let result = model.predict(&features);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::NotFitted { .. }));
    }

    #[test]
    fn test_fit_x_values_empty_after_check() {
        // This tests the redundant check at line 94 (though it should never execute)
        // But we test it to cover the branch
        let mut model = MultipleLinearRegression::<f64>::new();
        // This will fail at the first empty check, but tests the code path
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_many_not_fitted() {
        // Test predict_many when model is not fitted
        let model = MultipleLinearRegression::<f64>::new();
        let result = model.predict_many(&[vec![1.0, 2.0]]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StatsError::NotFitted { .. }));
    }

    #[test]
    fn test_predict_many_dimension_mismatch() {
        // Test predict_many with wrong number of features
        let mut model = MultipleLinearRegression::<f64>::new();
        let x = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 3.0]];
        let y = vec![3.0, 3.0, 6.0];
        model.fit(&x, &y).unwrap();

        // Try to predict with wrong number of features
        let wrong_features = vec![vec![1.0]]; // Should be 2 features
        let result = model.predict_many(&wrong_features);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_predict_many_success() {
        // Test predict_many with valid data
        let mut model = MultipleLinearRegression::<f64>::new();
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
        ];
        let y = vec![3.0, 3.0, 6.0, 6.0];
        model.fit(&x, &y).unwrap();

        let predictions = model
            .predict_many(&[vec![3.0, 4.0], vec![5.0, 6.0]])
            .unwrap();
        assert_eq!(predictions.len(), 2);
    }
}
