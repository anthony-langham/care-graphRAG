# TODO - Generated from plan-v001.md
Created: 2025-07-28

## Phase 1: Backend Deployment (Immediate Priority)

[X] **TASK-038**: Setup SST project for serverless deployment
  [X] Initialize SST project configuration
  [X] Configure eu-west-2 region
  [X] Setup project structure for Lambda functions

[X] **TASK-039**: Create Lambda functions (query, sync, health)
  [X] Create query Lambda handler
  [X] Create sync Lambda handler for scheduled updates
  [X] Create health check Lambda handler

[X] **TASK-040**: Setup Lambda layers with Python dependencies
  [X] Create requirements.txt for Lambda layer
  [X] Build Python dependencies layer
  [X] Configure layer deployment

[ ] **TASK-041**: Configure API Gateway with CORS
  [ ] Setup API Gateway routes
  [ ] Configure CORS headers for care.engineering
  [ ] Setup authentication (API keys/usage plans)

[ ] **TASK-042**: Setup environment config and secrets
  [ ] Configure MongoDB connection string in AWS Secrets Manager
  [ ] Configure OpenAI API key in AWS Secrets Manager
  [ ] Setup environment variables for Lambda functions

[ ] **TASK-043**: Implement Lambda handlers with FastAPI adapter
  [ ] Install and configure Mangum for FastAPI-Lambda adapter
  [ ] Adapt existing FastAPI endpoints for Lambda
  [ ] Ensure proper request/response handling

[ ] **TASK-044**: Configure Lambda settings (memory, timeout, concurrency)
  [ ] Set appropriate memory allocation (1024MB initial)
  [ ] Configure 30s timeout for query functions
  [ ] Configure 5min timeout for sync functions
  [ ] Set concurrency limits

[ ] **TASK-045**: Setup monitoring with CloudWatch and X-Ray
  [ ] Enable CloudWatch logging for all functions
  [ ] Configure X-Ray tracing for performance monitoring
  [ ] Setup CloudWatch alarms for errors and timeouts
  [ ] Create dashboard for key metrics

## Phase 2: Frontend Integration

[ ] Integrate care.engineering with new GraphRAG API endpoint
  [ ] Update frontend API configuration to point to new endpoint
  [ ] Implement API client for /query endpoint

[ ] Update frontend to call `/query` endpoint
  [ ] Implement request structure with question and max_sources
  [ ] Handle async API calls with appropriate loading states

[ ] Handle API responses and error states
  [ ] Parse and display answer content
  [ ] Handle 400, 408, and 500 error codes appropriately
  [ ] Display cost information if needed

[ ] Add clinical safety disclaimers and source attribution display
  [ ] Display sources with proper NICE guideline attribution
  [ ] Add clinical safety warning messages
  [ ] Show confidence scores where appropriate

## Phase 3: Testing and Validation

[ ] Test API endpoints
  [ ] Verify production deployment accessibility
  [ ] Test all error scenarios (400, 408, 500)
  [ ] Validate response structure and content

[ ] Clinical validation
  [ ] Test with real clinical scenarios
  [ ] Verify NICE guideline accuracy ≥ 90%
  [ ] Validate source attribution accuracy

[ ] Monitor performance
  [ ] Verify response times under 25 seconds
  [ ] Monitor cost per 100 queries (target < £0.30)
  [ ] Track CloudWatch metrics and alerts

## Success Criteria Checklist

[ ] GraphRAG API deployed and accessible
[ ] care.engineering successfully integrates with backend
[ ] Clinical questions answered with proper source attribution
[ ] Response times under 25 seconds
[ ] Cost per 100 queries under £0.30
[ ] NICE guideline accuracy ≥ 90%

## Notes

- Skip original Phase 10 tasks (TASK-036: Streamlit UI, TASK-037: React app) as care.engineering provides the frontend
- Focus on API-driven architecture with clean separation of concerns
- Leverage existing clinical UX patterns from care.engineering