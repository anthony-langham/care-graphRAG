# Plan v001: care.engineering Frontend Integration

## Overview

Integrate the existing live frontend at care.engineering with the Care-GraphRAG backend as an API service, eliminating the need for Phase 10 frontend development.

## Architecture

```
care.engineering (Existing Frontend)
    ↓ HTTPS/JSON API calls
API Gateway + Lambda (Care-GraphRAG Backend)  
    ↓ MongoDB connections
MongoDB Atlas (Graph + Vector Store)
```

## Benefits

1. **Existing Infrastructure**: Leverage live, tested frontend at care.engineering
2. **Separation of Concerns**: Clean API-driven architecture with clear boundaries
3. **Independent Scaling**: Backend (GraphRAG) and frontend can scale independently
4. **Cost Efficiency**: No need to build/maintain separate UI infrastructure
5. **Clinical Focus**: Existing site likely has appropriate clinical UX patterns

## Current Status

✅ **TASK-033 Complete**: QA endpoint with FastAPI integration
- `/query` endpoint with structured JSON responses
- CORS middleware configured for web frontend integration
- Comprehensive error handling (400, 408, 500 status codes)
- Health check endpoint with system information
- Input validation and timeout handling

## Implementation Plan

### Phase 1: Backend Deployment (Immediate Priority)
- **TASK-038**: Setup SST project for serverless deployment
- **TASK-039**: Create Lambda functions (query, sync, health)
- **TASK-040**: Setup Lambda layers with Python dependencies
- **TASK-041**: Configure API Gateway with CORS
- **TASK-042**: Setup environment config and secrets
- **TASK-043**: Implement Lambda handlers with FastAPI adapter
- **TASK-044**: Configure Lambda settings (memory, timeout, concurrency)
- **TASK-045**: Setup monitoring with CloudWatch and X-Ray

### Phase 2: Frontend Integration
- Integrate care.engineering with new GraphRAG API endpoint
- Update frontend to call `/query` endpoint
- Handle API responses and error states
- Add clinical safety disclaimers and source attribution display

### Phase 3: Skip Original Phase 10
- ~~TASK-036: Streamlit UI~~ (Not needed)
- ~~TASK-037: New React app~~ (Not needed)

## API Integration Details

### Endpoint Structure
```
POST /query
{
  "question": "What is first-line treatment for hypertension in a 45-year-old?",
  "max_sources": 5
}

Response:
{
  "answer": "For a 45-year-old patient with hypertension...",
  "sources": [...],
  "confidence": 0.85,
  "retrieval_method": "graph_primary",
  "cost_info": {...}
}
```

### Error Handling
- 400: Invalid input (question too long, invalid max_sources)
- 408: Timeout (query took longer than 25 seconds)
- 500: Internal server error (MongoDB connection, OpenAI API issues)

## Next Steps

1. **Deploy GraphRAG API**: Complete TASK-038 through TASK-045
2. **Test API endpoints**: Verify production deployment
3. **Frontend integration**: Update care.engineering to use new backend
4. **Clinical validation**: Test with real clinical scenarios
5. **Monitor performance**: CloudWatch metrics and cost tracking

## Success Criteria

- ✅ GraphRAG API deployed and accessible
- ✅ care.engineering successfully integrates with backend
- ✅ Clinical questions answered with proper source attribution
- ✅ Response times under 25 seconds
- ✅ Cost per 100 queries under £0.30
- ✅ NICE guideline accuracy ≥ 90%

---
Created: 2025-07-28 15:02:05 UTC
Status: Active