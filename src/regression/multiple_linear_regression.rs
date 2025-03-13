// src/regression/multiple_linear_regression.rs

use std::fmt::Debug;
use num_traits::{Float, NumCast};

/// Multiple linear regression model that fits a hyperplane to multivariate data points.
#[derive(Debug, Clone)]
pub struct MultipleLinearRegression<T = f64>
where
    T: Float + Debug + Default,
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

impl<T> MultipleLinearRegression<T>
where
    T: Float + Debug + Default + NumCast,
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
    /// * `Result<(), String>` - Ok if successful, Err with message if the inputs are invalid
    pub fn fit<U, V>(&mut self, x_values: &[Vec<U>], y_values: &[V]) -> Result<(), String>
    where
        U: NumCast + Copy,
        V: NumCast + Copy,
    {
        // Validate inputs
        if x_values.is_empty() || y_values.is_empty() {
            return Err("Cannot fit regression with empty arrays".to_string());
        }

        if x_values.len() != y_values.len() {
            return Err("Number of observations in X and Y must match".to_string());
        }

        self.n = x_values.len();
        
        // Check that all rows in x_values have the same length
        if x_values.is_empty() {
            return Err("X values array is empty".to_string());
        }
        
        self.p = x_values[0].len();
        
        for row in x_values {
            if row.len() != self.p {
                return Err("All rows in X must have the same number of features".to_string());
            }
        }

        // Convert input arrays to T type
        let mut x_cast: Vec<Vec<T>> = Vec::with_capacity(self.n);
        for row in x_values {
            let row_cast: Result<Vec<T>, String> = row
                .iter()
                .map(|&x| T::from(x).ok_or_else(|| "Failed to cast X value".to_string()))
                .collect();
            x_cast.push(row_cast?);
        }

        let y_cast: Vec<T> = y_values
            .iter()
            .map(|&y| T::from(y).ok_or_else(|| "Failed to cast Y value".to_string()))
            .collect::<Result<Vec<T>, String>>()?;

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
            },
            Err(e) => return Err(e),
        }

        // Calculate fitted values and R²
        let y_mean = y_cast.iter().fold(T::zero(), |acc, &y| acc + y) / T::from(self.n).unwrap();
        
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
                let n_minus_1 = T::from(self.n - 1).unwrap();
                let n_minus_p_minus_1 = T::from(self.n - self.p - 1).unwrap();
                
                self.adjusted_r_squared = T::one() - ((T::one() - self.r_squared) * 
                    n_minus_1 / n_minus_p_minus_1);
            }
        }
        
        // Calculate standard error
        if self.n > self.p + 1 {
            let n_minus_p_minus_1 = T::from(self.n - self.p - 1).unwrap();
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
        for i in 0..self.p {
            result = result + (self.coefficients[i + 1] * x[i]);
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
        let x_cast: Result<Vec<T>, ()> = x.iter()
            .map(|&val| T::from(val).ok_or(()))
            .collect();
            
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
        x_values.iter()
            .map(|x| self.predict(x))
            .collect()
    }
    
    // Helper function: Matrix multiplication where one matrix is transposed: A^T * B
    fn matrix_multiply_transpose(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
        let a_rows = a.len();
        let a_cols = if a_rows > 0 { a[0].len() } else { 0 };
        let b_rows = b.len();
        let b_cols = if b_rows > 0 { b[0].len() } else { 0 };
        
        // Result will be a_cols × b_cols
        let mut result = vec![vec![T::zero(); b_cols]; a_cols];
        
        for i in 0..a_cols {
            for j in 0..b_cols {
                let mut sum = T::zero();
                for k in 0..a_rows {
                    sum = sum + (a[k][i] * b[k][j]);
                }
                result[i][j] = sum;
            }
        }
        
        result
    }
    
    // Helper function: Multiply transposed matrix by vector: A^T * y
    fn vector_multiply_transpose(&self, a: &[Vec<T>], y: &[T]) -> Vec<T> {
        let a_rows = a.len();
        let a_cols = if a_rows > 0 { a[0].len() } else { 0 };
        
        let mut result = vec![T::zero(); a_cols];
        
        for i in 0..a_cols {
            let mut sum = T::zero();
            for j in 0..a_rows {
                sum = sum + (a[j][i] * y[j]);
            }
            result[i] = sum;
        }
        
        result
    }
    
    // Helper function: Solve a system of linear equations using Gaussian elimination
    fn solve_linear_system(&self, a: &[Vec<T>], b: &[T]) -> Result<Vec<T>, String> {
        let n = a.len();
        if n == 0 || a[0].len() != n || b.len() != n {
            return Err("Invalid matrix dimensions for linear system solving".to_string());
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
            
            for j in (i+1)..n {
                let abs_val = aug[j][i].abs();
                if abs_val > max_val {
                    max_row = j;
                    max_val = abs_val;
                }
            }
            
            let epsilon: T = T::from(1e-10).unwrap();
            if max_val < epsilon {
                return Err("Matrix is singular or near-singular".to_string());
            }
            
            // Swap rows if needed
            if max_row != i {
                aug.swap(i, max_row);
            }
            
            // Eliminate below
            for j in (i+1)..n {
                let factor = aug[j][i] / aug[i][i];
                
                for k in i..(n+1) {
                    aug[j][k] = aug[j][k] - (factor * aug[i][k]);
                }
            }
        }
        
        // Back substitution
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            
            for j in (i+1)..n {
                sum = sum - (aug[i][j] * x[j]);
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
        let x = vec![
            vec![1, 2],
            vec![2, 1],
            vec![3, 3],
            vec![4, 2],
        ];
        let y = vec![9, 8, 16, 15];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();
        
        // Test prediction: 1 + 2*5 + 3*4 = 1 + 10 + 12 = 23
        assert!(approx_equal(model.predict(&[5u32, 4u32]), 23.0, Some(1e-6)));
    }

    #[test]
    fn test_prediction_many() {
        let x = vec![
            vec![1, 2],
            vec![2, 1],
            vec![3, 3],
        ];
        let y = vec![9, 8, 16];

        let mut model = MultipleLinearRegression::<f64>::new();
        model.fit(&x, &y).unwrap();
        
        let new_x = vec![
            vec![1u32, 2u32],
            vec![5u32, 4u32],
        ];
        
        let predictions = model.predict_many(&new_x);
        assert_eq!(predictions.len(), 2);
        assert!(approx_equal(predictions[0], 9.0, Some(1e-6)));
        assert!(approx_equal(predictions[1], 23.0, Some(1e-6)));
    }
}