# Panic-Free Refactoring Plan

## Executive Summary

This document outlines a comprehensive plan to eliminate all panics from the `rs-stats` library and replace them with proper error handling using Rust's `Result` type. The goal is to make the library production-ready and panic-free.

**Target Version**: 2.0.0 (breaking changes expected)

**Estimated Effort**: High (affects all modules)

**Priority**: High (critical for production use)

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Error Type Design](#error-type-design)
3. [Refactoring Strategy](#refactoring-strategy)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Testing Strategy](#testing-strategy)
6. [Migration Path](#migration-path)
7. [Breaking Changes](#breaking-changes)
8. [Implementation Phases](#implementation-phases)

---

## Current State Analysis

### Panic Sources Identified

#### 1. Explicit `panic!` Calls (3 locations)
- `src/utils/combinatorics.rs`: `permutation()` and `combination()` - 2 panics
- `src/regression/decision_tree.rs`: Multiple panics for type conversions - ~5 panics

#### 2. `.unwrap()` Calls (95+ instances)
- Type conversions: `T::from(n).unwrap()` - ~30 instances
- Array indexing: Direct indexing without bounds checks - ~20 instances
- Option unwrapping: `node.value.as_ref().unwrap()` - ~15 instances
- Test code: `.unwrap()` in tests - ~30 instances (acceptable)

#### 3. `.expect()` Calls (4 instances)
- `src/hypothesis_tests/chi_square_test.rs`: Type conversions - 4 instances

#### 4. Unsafe Array Indexing
- Direct indexing: `features[idx][feature_idx]` - Multiple locations
- No bounds checking before access

### Current Error Handling Patterns

#### Good Patterns (Keep)
- `src/prob/*.rs`: Uses `Option<f64>` for error cases
- `src/hypothesis_tests/t_test.rs`: Uses `Result<TTestResult, &'static str>`
- `src/regression/linear_regression.rs`: Uses `Result<(), String>` for `fit()`

#### Patterns to Improve
- String-based errors (`Result<(), String>`) - should use structured error type
- `&'static str` errors - should use structured error type
- `Option<T>` for some cases - should use `Result<T, E>` for better error context

---

## Error Type Design

### Proposed Error Type

```rust
// src/error.rs
use thiserror::Error;

/// Main error type for the rs-stats library
#[derive(Error, Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Invalid input parameters
    #[error("Invalid input: {message}")]
    InvalidInput {
        message: String,
    },
    
    /// Type conversion failure
    #[error("Conversion error: {message}")]
    ConversionError {
        message: String,
    },
    
    /// Empty data provided when data is required
    #[error("Empty data: {message}")]
    EmptyData {
        message: String,
    },
    
    /// Dimension mismatch between arrays/vectors
    #[error("Dimension mismatch: {message}")]
    DimensionMismatch {
        message: String,
    },
    
    /// Numerical computation error (overflow, underflow, NaN, etc.)
    #[error("Numerical error: {message}")]
    NumericalError {
        message: String,
    },
    
    /// Model not fitted/trained before use
    #[error("Model not fitted: {message}")]
    NotFitted {
        message: String,
    },
    
    /// Invalid parameter value
    #[error("Invalid parameter: {message}")]
    InvalidParameter {
        message: String,
    },
    
    /// Index out of bounds
    #[error("Index out of bounds: {message}")]
    IndexOutOfBounds {
        message: String,
    },
    
    /// Division by zero or similar mathematical error
    #[error("Mathematical error: {message}")]
    MathematicalError {
        message: String,
    },
}

/// Convenience type alias for Result with StatsError
pub type StatsResult<T> = Result<T, StatsError>;

/// Helper macros for creating errors
#[macro_export]
macro_rules! invalid_input {
    ($($arg:tt)*) => {
        $crate::error::StatsError::InvalidInput {
            message: format!($($arg)*)
        }
    };
}

#[macro_export]
macro_rules! conversion_error {
    ($($arg:tt)*) => {
        $crate::error::StatsError::ConversionError {
            message: format!($($arg)*)
        }
    };
}
```

### Error Type Benefits

1. **Structured**: Each error variant has context
2. **Clone-able**: Can be stored and passed around
3. **Display**: Automatic `Display` implementation via `thiserror`
4. **Extensible**: Easy to add new error variants
5. **Type-safe**: Compiler ensures all errors are handled

---

## Refactoring Strategy

### Principles

1. **No Breaking Changes in v1.x**: Keep existing APIs working (deprecate, don't remove)
2. **Gradual Migration**: Introduce new error-returning APIs alongside old ones
3. **Backward Compatibility**: Provide migration helpers where possible
4. **Comprehensive Testing**: Ensure no regressions
5. **Documentation**: Update all docs with error handling examples

### Approach

1. **Phase 1**: Add error type and infrastructure
2. **Phase 2**: Fix critical panics (public APIs)
3. **Phase 3**: Fix internal panics
4. **Phase 4**: Update all APIs to use `Result`
5. **Phase 5**: Remove deprecated APIs (v2.0.0)

---

## File-by-File Breakdown

### Priority 1: Critical Public APIs

#### `src/utils/combinatorics.rs`
**Current Issues:**
- `permutation()`: Panics if `k > n`
- `combination()`: Panics if `k > n`

**Refactoring:**
```rust
// Before
pub fn permutation(n: u64, k: u64) -> u64 {
    if k > n {
        panic!("k cannot be greater than n");
    }
    // ...
}

// After
pub fn permutation(n: u64, k: u64) -> StatsResult<u64> {
    if k > n {
        return Err(StatsError::InvalidInput {
            message: format!("k ({}) cannot be greater than n ({})", k, n),
        });
    }
    Ok(((n - k + 1)..=n).product::<u64>())
}
```

**Breaking Change**: Yes (return type changes)
**Migration**: Provide `permutation_unchecked()` that panics for backward compat (deprecated)

---

#### `src/regression/decision_tree.rs`
**Current Issues:**
- Multiple `panic!` calls for type conversions (~5)
- `.unwrap()` calls for node access (~10)
- Array indexing without bounds checks

**Refactoring:**
```rust
// Before
let t_threshold = NumCast::from(threshold).unwrap_or_else(|| {
    panic!("Failed to convert threshold to the feature type");
});

// After
let t_threshold = NumCast::from(threshold).ok_or_else(|| {
    StatsError::ConversionError {
        message: "Failed to convert threshold to the feature type".to_string(),
    }
})?;
```

**Key Methods to Fix:**
- `fit()`: Already returns `Result`, but internal panics need fixing
- `predict()`: Should return `Result<T, StatsError>`
- `build_tree()`: Internal, but should propagate errors
- `find_best_split()`: Internal, but should return `Result`

**Breaking Change**: Partial (some methods already return `Result`)

---

#### `src/regression/linear_regression.rs`
**Current Issues:**
- `T::from(n).unwrap()` - ~5 instances
- Type conversions in `fit()` method

**Refactoring:**
```rust
// Before
let x_mean = x_cast.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(n).unwrap();

// After
let n_as_t = T::from(n).ok_or_else(|| {
    StatsError::ConversionError {
        message: format!("Failed to convert {} to type T", n),
    }
})?;
let x_mean = x_cast.iter().fold(T::zero(), |acc, &x| acc + x) / n_as_t;
```

**Breaking Change**: No (already returns `Result<(), String>`)

---

#### `src/regression/multiple_linear_regression.rs`
**Current Issues:**
- Similar to `linear_regression.rs`
- `T::from(n).unwrap()` - ~10 instances

**Refactoring**: Same pattern as `linear_regression.rs`

**Breaking Change**: No (already returns `Result`)

---

#### `src/hypothesis_tests/chi_square_test.rs`
**Current Issues:**
- `.expect()` calls for type conversions - 4 instances

**Refactoring:**
```rust
// Before
let obs = observed[i]
    .to_f64()
    .expect("Failed to convert observed value to f64");

// After
let obs = observed[i]
    .to_f64()
    .ok_or_else(|| StatsError::ConversionError {
        message: format!("Failed to convert observed value at index {} to f64", i),
    })?;
```

**Breaking Change**: Yes (return type changes from `Option` to `Result`)

---

### Priority 2: Internal Functions

#### `src/prob/*.rs` modules
**Current State:**
- Most use `Option<f64>` which is acceptable
- Some could benefit from `Result` for better error context

**Refactoring Strategy:**
- Keep `Option<f64>` for simple cases (empty data)
- Consider `Result<f64, StatsError>` for complex errors
- No breaking changes needed immediately

---

#### `src/hypothesis_tests/t_test.rs`
**Current State:**
- Already uses `Result<TTestResult, &'static str>`
- Should migrate to `Result<TTestResult, StatsError>`

**Refactoring:**
```rust
// Before
pub fn one_sample_t_test<T>(...) -> Result<TTestResult, &'static str>

// After
pub fn one_sample_t_test<T>(...) -> StatsResult<TTestResult>
```

**Breaking Change**: Yes (error type changes)

---

#### `src/hypothesis_tests/anova.rs`
**Current State:**
- Returns `Option<AnovaResult>`
- Should return `Result<AnovaResult, StatsError>`

**Refactoring:**
```rust
// Before
pub fn one_way_anova<T>(...) -> Option<AnovaResult>

// After
pub fn one_way_anova<T>(...) -> StatsResult<AnovaResult>
```

**Breaking Change**: Yes (return type changes)

---

### Priority 3: Helper Functions

#### Array Indexing
**Current Issues:**
- Direct indexing: `features[idx][feature_idx]`
- No bounds checking

**Refactoring Pattern:**
```rust
// Before
let value = features[idx][feature_idx];

// After
let row = features.get(idx).ok_or_else(|| {
    StatsError::IndexOutOfBounds {
        message: format!("Row index {} out of bounds (len: {})", idx, features.len()),
    }
})?;
let value = row.get(feature_idx).ok_or_else(|| {
    StatsError::IndexOutOfBounds {
        message: format!("Column index {} out of bounds (len: {})", feature_idx, row.len()),
    }
})?;
```

---

## Testing Strategy

### Unit Tests

1. **Error Cases**: Test all error conditions
   ```rust
   #[test]
   fn test_permutation_invalid_input() {
       assert!(matches!(
           permutation(5, 10),
           Err(StatsError::InvalidInput { .. })
       ));
   }
   ```

2. **Success Cases**: Ensure existing functionality still works
   ```rust
   #[test]
   fn test_permutation_valid() {
       assert_eq!(permutation(5, 3).unwrap(), 60);
   }
   ```

3. **Edge Cases**: Test boundary conditions
   ```rust
   #[test]
   fn test_permutation_edge_cases() {
       assert_eq!(permutation(5, 5).unwrap(), 120);
       assert_eq!(permutation(5, 0).unwrap(), 1);
   }
   ```

### Integration Tests

1. **End-to-End**: Test complete workflows with error handling
2. **Error Propagation**: Ensure errors propagate correctly through call chains

### Regression Tests

1. **No Panics**: Add tests that verify no panics occur
   ```rust
   #[test]
   fn test_no_panics_on_invalid_input() {
       // This should not panic
       let _ = permutation(5, 10);
   }
   ```

---

## Migration Path

### Phase 1: Infrastructure (v1.3.0)
- [ ] Add `thiserror` dependency
- [ ] Create `src/error.rs` with `StatsError`
- [ ] Add error type to public API
- [ ] Create helper macros
- [ ] Update documentation

### Phase 2: Critical Fixes (v1.4.0)
- [ ] Fix `combinatorics.rs` panics
- [ ] Fix `decision_tree.rs` critical panics
- [ ] Fix `chi_square_test.rs` panics
- [ ] Add comprehensive tests

### Phase 3: Internal Refactoring (v1.5.0)
- [ ] Fix all `.unwrap()` calls in public APIs
- [ ] Fix array indexing issues
- [ ] Update error types from `String` to `StatsError`
- [ ] Add bounds checking everywhere

### Phase 4: API Standardization (v2.0.0)
- [ ] Migrate all `Option<T>` to `Result<T, StatsError>` where appropriate
- [ ] Migrate all `Result<T, String>` to `StatsResult<T>`
- [ ] Remove deprecated APIs
- [ ] Update all documentation
- [ ] Create migration guide

### Backward Compatibility

For v1.x releases, provide deprecated wrapper functions:

```rust
#[deprecated(note = "Use permutation() which returns Result")]
pub fn permutation_unchecked(n: u64, k: u64) -> u64 {
    permutation(n, k).expect("Invalid parameters")
}
```

---

## Breaking Changes

### Summary

| Module | Breaking Changes | Version |
|--------|-----------------|---------|
| `utils::combinatorics` | Return type: `u64` → `StatsResult<u64>` | 2.0.0 |
| `hypothesis_tests::chi_square_test` | Return type: `Option<T>` → `StatsResult<T>` | 2.0.0 |
| `hypothesis_tests::anova` | Return type: `Option<T>` → `StatsResult<T>` | 2.0.0 |
| `hypothesis_tests::t_test` | Error type: `&str` → `StatsError` | 2.0.0 |
| `regression::decision_tree` | Some methods: `T` → `StatsResult<T>` | 2.0.0 |

### Migration Guide Template

```markdown
## Migrating from v1.x to v2.0.0

### Combinatorics Functions

**Before:**
```rust
let result = permutation(5, 3);
```

**After:**
```rust
let result = permutation(5, 3)?;  // or handle error
match permutation(5, 3) {
    Ok(value) => println!("{}", value),
    Err(e) => eprintln!("Error: {}", e),
}
```
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Set up error handling infrastructure

- [ ] Add `thiserror = "1.0"` to `Cargo.toml`
- [ ] Create `src/error.rs`
- [ ] Export error type from `src/lib.rs`
- [ ] Write comprehensive error type tests
- [ ] Update `CHANGELOG.md`

**Deliverables:**
- Error type ready for use
- Documentation updated
- Tests passing

---

### Phase 2: Critical Panics (Week 2)
**Goal**: Fix all explicit panics in public APIs

- [ ] Fix `combinatorics.rs` (2 panics)
- [ ] Fix `decision_tree.rs` public methods (5 panics)
- [ ] Fix `chi_square_test.rs` (4 panics)
- [ ] Add tests for all error cases
- [ ] Update examples in documentation

**Deliverables:**
- No panics in public APIs
- All error cases tested
- Examples updated

---

### Phase 3: Internal Refactoring (Week 3-4)
**Goal**: Fix all `.unwrap()` and unsafe indexing

- [ ] Fix type conversions in regression modules
- [ ] Add bounds checking for all array access
- [ ] Fix internal helper functions
- [ ] Update error messages to be more descriptive
- [ ] Add error context where helpful

**Deliverables:**
- No `.unwrap()` in production code
- All array access is bounds-checked
- Better error messages

---

### Phase 4: API Standardization (Week 5-6)
**Goal**: Standardize all APIs to use `StatsResult`

- [ ] Migrate `Option<T>` to `StatsResult<T>` where appropriate
- [ ] Migrate `Result<T, String>` to `StatsResult<T>`
- [ ] Migrate `Result<T, &str>` to `StatsResult<T>`
- [ ] Update all documentation
- [ ] Create migration guide

**Deliverables:**
- Consistent error handling across library
- Complete documentation
- Migration guide ready

---

### Phase 5: Cleanup (Week 7)
**Goal**: Final polish and release preparation

- [ ] Remove deprecated functions (v2.0.0)
- [ ] Final documentation review
- [ ] Performance testing (ensure no regressions)
- [ ] Update version to 2.0.0
- [ ] Release notes

**Deliverables:**
- v2.0.0 ready for release
- All tests passing
- Documentation complete

---

## Success Criteria

### Must Have
- [ ] Zero panics in public APIs
- [ ] All errors use `StatsError` type
- [ ] Comprehensive test coverage for error cases
- [ ] Documentation updated with error handling examples
- [ ] Migration guide available

### Nice to Have
- [ ] Error context helpers (e.g., `.context()` method)
- [ ] Error recovery suggestions in error messages
- [ ] Performance benchmarks show no regression
- [ ] Backward compatibility shims for v1.x

---

## Risk Assessment

### High Risk
- **Breaking Changes**: Users need to update code
  - **Mitigation**: Provide migration guide and deprecated wrappers
- **Performance**: Error handling adds overhead
  - **Mitigation**: Benchmark and optimize hot paths

### Medium Risk
- **Test Coverage**: Need comprehensive error case tests
  - **Mitigation**: Systematic test writing, use coverage tools
- **Documentation**: Large amount of documentation to update
  - **Mitigation**: Incremental updates, prioritize public APIs

### Low Risk
- **Internal Refactoring**: Less visible to users
  - **Mitigation**: Thorough testing, gradual rollout

---

## Dependencies

### New Dependencies
```toml
[dependencies]
thiserror = "1.0"  # For error type derivation
```

### No Breaking Dependency Changes
- All existing dependencies remain compatible

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Foundation | 1 week | TBD | TBD |
| Phase 2: Critical Panics | 1 week | TBD | TBD |
| Phase 3: Internal Refactoring | 2 weeks | TBD | TBD |
| Phase 4: API Standardization | 2 weeks | TBD | TBD |
| Phase 5: Cleanup | 1 week | TBD | TBD |
| **Total** | **7 weeks** | | |

---

## Notes

- This is a major refactoring that will require careful planning
- Consider creating a feature branch for the entire refactoring
- Regular testing and validation at each phase
- Consider user feedback during beta releases
- Performance testing is critical to ensure no regressions

---

## References

- [Rust Error Handling Best Practices](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [thiserror Documentation](https://docs.rs/thiserror/)
- [Result Type Documentation](https://doc.rust-lang.org/std/result/)

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-17  
**Status**: Draft - Ready for Review


