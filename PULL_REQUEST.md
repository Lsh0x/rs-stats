# Add Decision Tree Implementation for Regression and Classification

## Overview
This PR adds a generic implementation of Decision Trees that supports both regression and classification tasks. The implementation is designed to be flexible, supporting various data types through generic parameters and trait bounds.

## Key Features
- Support for both regression and classification tasks
- Generic implementation that works with various numeric types
- Multiple split criteria:
  - Mean Squared Error (MSE) and Mean Absolute Error (MAE) for regression
  - Gini impurity and Entropy for classification
- Parallel processing using Rayon for performance optimization
- Comprehensive tree inspection and visualization features
- Configurable parameters:
  - Maximum tree depth
  - Minimum samples for split
  - Minimum samples per leaf

## Technical Implementation
- Generic type parameters `T` and `F` for input features and floating-point calculations
- Trait bounds ensuring type safety and required operations
- Thread-safe implementation with Send + Sync bounds
- Custom implementations for special data types (e.g., Duration, TestFloat)
- Comprehensive error handling and edge cases

## Test Coverage
The implementation includes extensive tests covering:
- Medical diagnosis prediction
- System failure prediction
- Security incident classification
- Custom data type handling (Duration-based performance analysis)
- Edge cases and error conditions

## Dependencies Added
- Added `rayon = "1.8"` for parallel processing capabilities

## Usage Example
```rust
let mut tree = DecisionTree::<f64, f64>::new(
    TreeType::Regression,
    SplitCriterion::Mse,
    5,   // max_depth
    2,   // min_samples_split
    1    // min_samples_leaf
);

// Train the model
tree.fit(&features, &target);

// Make predictions
let predictions = tree.predict(&test_features);
```

