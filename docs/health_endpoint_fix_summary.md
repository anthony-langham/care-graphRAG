# Health Endpoint Refactoring Summary

## 1. Problem Diagnosis

*   The `/health` endpoint, located at `functions/handlers/health.py`, currently establishes a direct and isolated `pymongo` connection to verify the MongoDB status.
*   This implementation bypasses the application's standardized `graphrag` module, which contains a centralized `MongoDBClient` in `functions/graphrag/mongo_client.py`.
*   Although the `/health` and `/test-mongodb` endpoints report success (confirming correct credentials and network access), they fail to validate whether the `graphrag` module can be imported correctly by other Lambda functions. This discrepancy creates a misleading and incomplete health status.

## 2. Proposed Solution

I will refactor `functions/handlers/health.py` to utilize the shared `MongoDBClient` for its connection test. The plan is as follows:

1.  **Remove Direct `pymongo` Logic**: Eliminate the manual `pymongo` connection code from the `health_check` function.
2.  **Import Centralized Client**: Add the line `from graphrag.mongo_client import get_mongo_client` to the top of `health.py`.
3.  **Use Shared Health Check**: Replace the direct connection test with a call to the centralized health check method: `get_mongo_client().health_check()`.

## 3. Benefits

*   **Consistent Connection Handling**: This change ensures that all components of the application connect to MongoDB using the same robust, centralized logic.
*   **Accurate Health Checks**: A successful health check will now serve as a true end-to-end validation, confirming both that the `graphrag` module is importable and that the database is reachable.
*   **Improved Maintainability**: Centralizing the database connection logic simplifies future updates, debugging, and maintenance.
