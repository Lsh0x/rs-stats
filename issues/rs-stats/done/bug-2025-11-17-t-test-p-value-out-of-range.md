# Fix t-test p-value calculation producing values > 1.0

**Type:** bug  
**Severity:** high  
**Status:** done  
**Branch:** fix/hypothesis-tests-t-test-p-value-range  
**Linked roadmap section:** N/A

---

## 🧪 Reproduction
1. Run any t-test (one-sample, two-sample, or paired)
2. Check the returned p_value in TTestResult
3. Observe that p_value can be greater than 1.0 (e.g., between 1.0 and 2.0)

**Expected:** p_value should always be between 0.0 and 1.0 (inclusive)

**Actual:** p_value can be between 1.0 and 2.0

**Logs/Artifacts:** 
- Location: `src/hypothesis_tests/t_test.rs`, lines 377-381
- Current code:
```rust
let a = df / (df + t_stat * t_stat);
let ix = 0.5 * incomplete_beta(0.5 * df, 0.5, a);
// Two-tailed p-value
2.0 * (1.0 - ix)
```

## 🧷 Suspected Root Cause
The `0.5` multiplier on `incomplete_beta` is incorrect. Since `incomplete_beta` returns a value in [0, 1]:
- `ix = 0.5 * [0, 1] = [0, 0.5]`
- `1.0 - ix = [0.5, 1.0]`
- `2.0 * (1.0 - ix) = [1.0, 2.0]` ❌

The fix should remove the `0.5` multiplier:
```rust
let ix = incomplete_beta(0.5 * df, 0.5, a);
2.0 * (1.0 - ix)
```

This would give:
- `ix = [0, 1]`
- `1.0 - ix = [0, 1]`
- `2.0 * (1.0 - ix) = [0, 2.0]` ❌ Still wrong!

Actually, wait - if `incomplete_beta(0.5 * df, 0.5, a)` gives us P(T ≤ |t|), then for two-tailed:
- P(|T| > |t|) = 2 * (1 - P(T ≤ |t|)) = 2 * (1 - incomplete_beta(...))

But we need to verify what `incomplete_beta` actually returns. Looking at the t-distribution formula:
- The regularized incomplete beta I_x(a, b) where x = df/(df + t^2) gives the CDF
- For two-tailed: 2 * (1 - I_x(df/2, 0.5))

So the correct formula should be:
```rust
let ix = incomplete_beta(0.5 * df, 0.5, a);
2.0 * (1.0 - ix)
```

But we need to ensure this doesn't exceed 1.0. Actually, if `ix` represents P(T ≤ |t|), then:
- For small t: ix ≈ 0.5, so 2 * (1 - 0.5) = 1.0 ✓
- For large t: ix ≈ 1.0, so 2 * (1 - 1.0) = 0.0 ✓
- For t = 0: ix ≈ 0.5, so 2 * (1 - 0.5) = 1.0 ✓

So removing the 0.5 multiplier should fix it, but we should also add a clamp to ensure p_value is in [0, 1].

## ✅ Acceptance Criteria
- [ ] p_value is always in the range [0.0, 1.0]
- [ ] All existing t-test tests pass
- [ ] Add regression test that verifies p_value range
- [ ] Verify p-values match expected statistical values for known test cases
- [ ] CHANGELOG entry added

## 🔧 Fix Plan
1. Create/switch to branch `fix/hypothesis-tests-t-test-p-value-range`
2. Write failing regression test for p-value range validation
3. Fix `calculate_p_value` function by removing the `0.5` multiplier
4. Add clamp to ensure p_value is in [0.0, 1.0] range
5. Verify fix with statistical test cases
6. Update docs if needed
7. Move issue to `in_progress/` then `done/`
8. Create PR referencing this issue

## 🧯 Rollback/Feature Flag
- Direct fix, no feature flag needed
- Can rollback by reverting the commit

## 🔗 Discussion Notes
The bug was discovered when user noticed p-values could exceed 1.0, which is mathematically invalid. P-values represent probabilities and must be in [0, 1].

