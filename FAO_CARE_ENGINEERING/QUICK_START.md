# Quick Start Guide - Get Running in 10 Minutes

**For:** care.engineering developers  
**Goal:** Get the GraphRAG integration working in your local environment  
**Time:** ~10 minutes  

---

## 🚀 10-Minute Setup

### Step 1: Test API Connectivity (2 minutes)

```bash
# Test the staging API is working
curl https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com/health

# Expected response:
# {"status":"healthy","service":"nice-graphrag",...}

# Test a query
curl -X POST https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is hypertension?", "max_tokens": 1000}'

# Expected: JSON response with answer and sources
```

✅ **Success**: You see JSON responses? Great! The API is working.  
❌ **Issues**: Check your internet connection or contact backend team.

### Step 2: Environment Setup (2 minutes)

```bash
# Add to your .env.local file
echo "NEXT_PUBLIC_GRAPHRAG_API_URL=https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com" >> .env.local
echo "GRAPHRAG_ENVIRONMENT=development" >> .env.local

# Verify
cat .env.local | grep GRAPHRAG
```

### Step 3: Create Basic API Client (3 minutes)

Create `src/services/graphrag-api.ts`:

```typescript
interface GraphRAGRequest {
  question: string;
  max_tokens?: number;
}

interface GraphRAGResponse {
  answer: string;
  sources: { source: string; content: string }[];
  metadata: any;
}

class GraphRAGClient {
  private baseUrl = process.env.NEXT_PUBLIC_GRAPHRAG_API_URL!;

  async query(request: GraphRAGRequest): Promise<GraphRAGResponse> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.status}`);
    }

    return response.json();
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

export const graphragClient = new GraphRAGClient();
```

### Step 4: Create Simple Component (3 minutes)

Create `src/components/SimpleGraphRAGTest.tsx`:

```tsx
import React, { useState } from 'react';
import { graphragClient } from '../services/graphrag-api';

export const SimpleGraphRAGTest: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const result = await graphragClient.query({
        question,
        max_tokens: 1000
      });
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '20px auto', padding: '20px' }}>
      <h2>GraphRAG Test</h2>
      
      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a clinical question..."
          rows={3}
          style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          style={{ padding: '10px 20px', background: '#0066cc', color: 'white', border: 'none' }}
        >
          {loading ? 'Searching...' : 'Ask Question'}
        </button>
      </form>

      {error && (
        <div style={{ background: '#ffe6e6', padding: '10px', margin: '10px 0', borderRadius: '4px' }}>
          Error: {error}
        </div>
      )}

      {response && (
        <div style={{ background: '#f8f9fa', padding: '15px', margin: '15px 0', borderRadius: '4px' }}>
          <h3>Answer:</h3>
          <p>{response.answer}</p>
          
          <h4>Sources:</h4>
          {response.sources.map((source: any, i: number) => (
            <div key={i} style={{ background: 'white', padding: '8px', margin: '5px 0', borderRadius: '3px' }}>
              <strong>{source.source}</strong>: {source.content}
            </div>
          ))}
          
          <small>
            Environment: {response.metadata?.deployment_stage} | 
            Handler: {response.metadata?.handler_type}
          </small>
        </div>
      )}
    </div>
  );
};
```

### Step 5: Add to Your App & Test (1 minute)

Add the component to your app:

```tsx
// In your main component or page
import { SimpleGraphRAGTest } from '../components/SimpleGraphRAGTest';

export default function TestPage() {
  return (
    <div>
      <h1>GraphRAG Integration Test</h1>
      <SimpleGraphRAGTest />
    </div>
  );
}
```

Start your dev server:
```bash
npm run dev
```

---

## ✅ Quick Test

1. **Open your app** in the browser
2. **Ask a question**: "What is hypertension?"
3. **Expected result**: You should see an answer and sources

**Success**: You're connected to the GraphRAG API! 🎉  
**Issues**: Check the troubleshooting section below.

---

## 🔍 Quick Troubleshooting

### Issue: "Failed to fetch" Error
**Cause**: CORS or network issue  
**Fix**: 
```typescript
// Add Origin header to your API client
headers: {
  'Content-Type': 'application/json',
  'Origin': window.location.origin, // Add this line
}
```

### Issue: Environment Variable Not Found
**Cause**: .env.local not loaded  
**Fix**:
```bash
# Restart your dev server
npm run dev

# Check the variable is loaded
console.log(process.env.NEXT_PUBLIC_GRAPHRAG_API_URL);
```

### Issue: 404 Not Found
**Cause**: Wrong API URL  
**Fix**: Double-check the URL:
```
https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com
```

### Issue: Timeout Errors
**Cause**: Slow API response  
**Fix**: The staging API should respond quickly. If you get timeouts:
1. Check your internet connection
2. Try the curl command again
3. Contact backend team if persistent

---

## 🎯 What You Should See

### Successful Health Check:
```json
{
  "status": "healthy",
  "service": "nice-graphrag",
  "version": "1.0.0",
  "deployment_stage": "staging",
  "environment_check": {
    "mongodb_uri_configured": false,
    "openai_key_configured": false,
    "environment": "dev",
    "sst_version": "v3"
  }
}
```

### Successful Query Response:
```json
{
  "answer": "This is a minimal deployment test response. Full GraphRAG integration will be added after successful staging deployment.",
  "sources": [
    {
      "source": "deployment_test",
      "content": "minimal handler"
    }
  ],
  "metadata": {
    "deployment_stage": "staging",
    "handler_type": "minimal",
    "mongodb_configured": false,
    "openai_configured": false,
    "sst_version": "v3"
  }
}
```

**Note**: The response is currently a placeholder. Full NICE clinical data will be available when the backend team completes full integration.

---

## 📝 Next Steps

Once you have the basic integration working:

1. **Read the full documentation**:
   - `care-engineering-frontend.md` - Complete task specifications
   - `API_EXAMPLES.md` - Production-ready code examples
   - `TODO.md` - Your task list (TASK-201 to TASK-207)

2. **Start development**:
   - Begin with TASK-201 (API Client Implementation)
   - Follow the task breakdown in the TODO.md file
   - Use the code examples as templates

3. **Set up proper testing**:
   - Add unit tests for your API client
   - Set up integration tests
   - Configure error handling

---

## 🆘 Need Help?

### Quick References
- **API Base URL**: `https://w46s2t96h8.execute-api.eu-west-2.amazonaws.com`
- **Health Endpoint**: `GET /health`
- **Query Endpoint**: `POST /query`
- **CORS**: Enabled for care.engineering domains

### Support
- **Documentation**: All files in this FAO_CARE_ENGINEERING folder
- **Issues**: Create GitHub issue in care-graphRAG repository
- **API Problems**: Test with curl first, then contact backend team

---

**Congratulations! You now have a working GraphRAG integration. Time to build something amazing!** 🚀

---

*Quick Start Guide*  
*Estimated Setup Time: 10 minutes*  
*API Status: ✅ Ready*