# {{title}}

**Type:** bug  
**Severity:** low | medium | high | critical  
**Status:** todo  
**Branch:** fix/{{scope}}-{{slug}}  
**Linked roadmap section:** {{phase}}

---

## 🧪 Reproduction
1. {{step}}
2. {{step}}

**Expected:** {{expected}}

**Actual:** {{actual}}

**Logs/Artifacts:** {{links or snippets}}

## 🧷 Suspected Root Cause
{{hypothesis}}

## ✅ Acceptance Criteria
- [ ] Repro steps pass with fix
- [ ] Regression test added to guard against recurrence
- [ ] Related docs updated (troubleshooting, known issues if relevant)
- [ ] CHANGELOG entry added

## 🔧 Fix Plan
1. Create/switch to branch `fix/{{scope}}-{{slug}}`
2. Write failing regression test
3. Implement fix with minimal surface area
4. Add guardrails (invariants, input validation)
5. Update docs
6. Move issue to `in_progress/` then `done/`
7. Create PR referencing this issue

## 🧯 Rollback/Feature Flag
- {{strategy}}

## 🔗 Discussion Notes
{{from exploration}}


