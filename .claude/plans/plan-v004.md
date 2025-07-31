# Plan v004

## Plan to Integrate Full GraphRAG System into Production Lambda

### Current Status:
- Production API is deployed and working with authentication and rate limiting
- Frontend is successfully connecting and getting placeholder responses
- The full GraphRAG system (hybrid retriever, QA chain) exists in the main codebase but isn't integrated into Lambda

### Plan Overview:

#### 1. Create Lambda-Compatible GraphRAG Module (New files in functions/src/)
- Copy essential GraphRAG components into Lambda function directory
- Create simplified versions without heavy dependencies (NumPy, Pandas, etc.)
- Components needed:
  - `functions/src/graphrag/mongo_client.py` - Lambda-optimized MongoDB connection
  - `functions/src/graphrag/hybrid_retriever.py` - Core retrieval logic
  - `functions/src/graphrag/qa_chain.py` - Question answering logic
  - `functions/src/graphrag/config.py` - Simplified settings for Lambda

#### 2. Update Lambda Dependencies (functions/pyproject.toml)
Add required LangChain dependencies:
- langchain==0.3.26
- langchain-openai==0.3.28
- langchain-mongodb==0.6.2
- tenacity==8.2.3

#### 3. Update Production Handler (functions/src/functions/query_prod.py)
- Import the Lambda-compatible GraphRAG components
- Initialize QA chain with MongoDB connection
- Replace placeholder response with actual GraphRAG query processing
- Add proper error handling for MongoDB connection issues
- Implement response caching for repeated queries

#### 4. Environment Variable Updates
- Ensure MongoDB URI is accessible via SST secrets
- Add MongoDB database and collection names to Lambda environment
- Configure OpenAI API key access

#### 5. Testing & Validation
- Test MongoDB connection from Lambda
- Verify GraphRAG query processing
- Check response times and optimize if needed
- Validate clinical accuracy of responses

### Implementation Steps:

1. **Create Lambda-compatible GraphRAG modules** with minimal dependencies
2. **Update Lambda dependencies** to include LangChain packages
3. **Modify production handler** to use real GraphRAG
4. **Deploy to production** with updated configuration
5. **Test end-to-end** with frontend team

### Expected Outcome:
- Real clinical answers from NICE CKS data instead of placeholders
- Proper source attribution from the knowledge graph
- Maintained performance (< 5 second response time)
- Full audit trail of queries and responses

---
Created: 2025-01-31T18:50:00Z
Status: Active