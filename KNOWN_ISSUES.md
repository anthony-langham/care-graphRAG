# Known Issues

## Pre-existing Test Failures (Not Related to Hybrid Retrieval Implementation)

### test_graph_persistence.py - 3 Failing Tests

**Status**: These failures existed before TASK-023 hybrid retrieval implementation and are not caused by our changes.

1. **`test_domain_coverage_calculation`**
   - **Error**: `AssertionError: 24 != 17`
   - **Issue**: Expected 17 entity types but found 24
   - **Likely Cause**: VALID_ENTITY_TYPES list was expanded but test wasn't updated

2. **`test_get_graph_statistics_with_data`**
   - **Error**: `AssertionError: 3 != 2` 
   - **Issue**: Expected 2 total nodes but found 3
   - **Likely Cause**: Test data setup creates more nodes than expected

3. **`test_graph_builder_initialization`**
   - **Error**: Connection string mismatch
   - **Expected**: `mongodb://localhost:27017`
   - **Actual**: Real MongoDB Atlas connection string
   - **Likely Cause**: Test is picking up real config instead of mock

### Impact
- These failures don't affect the hybrid retrieval functionality
- Core system works correctly (32 other tests pass)
- Integration tests confirm hybrid retriever is operational

### Recommendation
- Fix these tests in a separate task focused on graph persistence testing
- Update entity type counts and mock connection strings properly
- Not blocking for hybrid retrieval deployment

---

## Current Working Status
- ✅ Hybrid Retrieval: Fully functional (13/13 tests pass)
- ✅ Core System: Working (32/35 tests pass)  
- ❌ Graph persistence tests need fixing (separate from hybrid retrieval)

Last updated: 2025-07-25