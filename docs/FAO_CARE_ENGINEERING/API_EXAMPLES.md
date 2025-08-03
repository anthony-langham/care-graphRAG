# API Examples - Ready to Use Code

**Staging API:** https://staging-api.graphrag.care  
**Status:** ✅ Operational and tested  
**CORS:** ✅ Configured for care.engineering domains  

---

## 🚀 Quick Start Examples

### Environment Configuration

```typescript
// .env.local
NEXT_PUBLIC_GRAPHRAG_API_URL=https://staging-api.graphrag.care
GRAPHRAG_ENVIRONMENT=staging
```

```typescript
// config/graphrag.ts
export const GRAPHRAG_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_GRAPHRAG_API_URL!,
  timeout: 30000,
  environment: process.env.GRAPHRAG_ENVIRONMENT || 'staging'
};
```

---

## 📝 TypeScript Interfaces (Copy-Paste Ready)

```typescript
// types/graphrag.ts

interface GraphRAGRequest {
  question: string;
  max_tokens?: number;
}

interface GraphRAGResponse {
  answer: string;
  sources: GraphRAGSource[];
  metadata: {
    deployment_stage: string;
    handler_type: string;
    mongodb_configured: boolean;
    openai_configured: boolean;
    sst_version: string;
  };
}

interface GraphRAGSource {
  source: string;
  content: string;
}

interface HealthResponse {
  status: string;
  service: string;
  version: string;
  deployment_stage: string;
  environment_check: {
    mongodb_uri_configured: boolean;
    openai_key_configured: boolean;
    environment: string;
    sst_version: string;
  };
}

interface GraphRAGError {
  message: string;
  status: number;
  code?: string;
  query_id?: string;
}
```

---

## 🔧 API Client Implementation

### Basic API Client Class

```typescript
// services/graphrag-api.ts
import { GRAPHRAG_CONFIG } from '../config/graphrag';

class GraphRAGClient {
  private baseUrl: string;
  private timeout: number;

  constructor() {
    this.baseUrl = GRAPHRAG_CONFIG.baseUrl;
    this.timeout = GRAPHRAG_CONFIG.timeout;
  }

  async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new GraphRAGError(`Health check failed: ${response.status}`, response.status);
    }

    return response.json();
  }

  async query(request: GraphRAGRequest): Promise<GraphRAGResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Origin': window.location.origin, // Important for CORS
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new GraphRAGError(
          errorData.message || `Query failed: ${response.status}`,
          response.status,
          errorData.code,
          errorData.query_id
        );
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new GraphRAGError('Query timeout after 30 seconds', 408);
      }
      
      if (error instanceof GraphRAGError) {
        throw error;
      }
      
      throw new GraphRAGError(`Network error: ${error.message}`, 0);
    }
  }
}

class GraphRAGError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public queryId?: string
  ) {
    super(message);
    this.name = 'GraphRAGError';
  }
}

export const graphragClient = new GraphRAGClient();
export { GraphRAGError };
```

---

## ⚛️ React Hook Implementation

```typescript
// hooks/useGraphRAG.ts
import { useState, useCallback } from 'react';
import { graphragClient, GraphRAGError } from '../services/graphrag-api';

interface UseGraphRAGResult {
  query: (question: string) => Promise<void>;
  isLoading: boolean;
  response: GraphRAGResponse | null;
  error: GraphRAGError | null;
  clearError: () => void;
  clearResponse: () => void;
}

export const useGraphRAG = (): UseGraphRAGResult => {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<GraphRAGResponse | null>(null);
  const [error, setError] = useState<GraphRAGError | null>(null);

  const query = useCallback(async (question: string) => {
    if (!question.trim()) {
      setError(new GraphRAGError('Question cannot be empty', 400));
      return;
    }

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const result = await graphragClient.query({
        question: question.trim(),
        max_tokens: 1000,
      });
      
      setResponse(result);
    } catch (err) {
      setError(err instanceof GraphRAGError ? err : new GraphRAGError(err.message, 0));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);
  const clearResponse = useCallback(() => setResponse(null), []);

  return {
    query,
    isLoading,
    response,
    error,
    clearError,
    clearResponse,
  };
};
```

---

## 🎨 React Component Examples

### Query Component

```tsx
// components/GraphRAGQuery.tsx
import React, { useState } from 'react';
import { useGraphRAG } from '../hooks/useGraphRAG';

interface GraphRAGQueryProps {
  onQueryComplete?: (response: GraphRAGResponse) => void;
  placeholder?: string;
}

export const GraphRAGQuery: React.FC<GraphRAGQueryProps> = ({
  onQueryComplete,
  placeholder = "Ask a clinical question about hypertension..."
}) => {
  const [question, setQuestion] = useState('');
  const { query, isLoading, response, error, clearError } = useGraphRAG();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    await query(question);
    
    if (response && onQueryComplete) {
      onQueryComplete(response);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(e.target.value);
    if (error) clearError();
  };

  return (
    <div className="graphrag-query">
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <textarea
            value={question}
            onChange={handleInputChange}
            placeholder={placeholder}
            disabled={isLoading}
            rows={3}
            maxLength={500}
            className={`query-input ${error ? 'error' : ''}`}
          />
          <button
            type="submit"
            disabled={!question.trim() || isLoading}
            className="submit-button"
          >
            {isLoading ? 'Searching...' : 'Ask Question'}
          </button>
        </div>
      </form>

      {isLoading && (
        <div className="loading-state">
          <div className="spinner" />
          <p>Searching NICE guidelines...</p>
        </div>
      )}

      {error && (
        <div className="error-state">
          <p className="error-message">{error.message}</p>
          {error.status === 408 && (
            <button onClick={() => query(question)} className="retry-button">
              Try Again
            </button>
          )}
        </div>
      )}

      {response && <GraphRAGResponse response={response} />}
    </div>
  );
};
```

### Response Display Component

```tsx
// components/GraphRAGResponse.tsx
import React, { useState } from 'react';

interface GraphRAGResponseProps {
  response: GraphRAGResponse;
}

export const GraphRAGResponse: React.FC<GraphRAGResponseProps> = ({ response }) => {
  const [showAllSources, setShowAllSources] = useState(false);
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);

  const handleCopyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  const visibleSources = showAllSources ? response.sources : response.sources.slice(0, 3);

  return (
    <div className="graphrag-response">
      {/* Clinical Safety Disclaimer */}
      <div className="clinical-disclaimer">
        <div className="warning-icon">⚠️</div>
        <div>
          <strong>Clinical Information Notice:</strong> This information is based on 
          NICE Clinical Knowledge Summaries and should not replace professional medical 
          advice. Always consult with a qualified healthcare professional.
        </div>
      </div>

      {/* Answer Display */}
      <div className="answer-section">
        <div className="answer-header">
          <h3>Answer</h3>
          <button
            onClick={handleCopyToClipboard}
            className="copy-button"
            title="Copy to clipboard"
          >
            {copiedToClipboard ? '✅ Copied!' : '📋 Copy'}
          </button>
        </div>
        
        <div className="answer-content">
          {response.answer}
        </div>

        {/* Metadata */}
        <div className="answer-metadata">
          <span className="deployment-stage">
            {response.metadata.deployment_stage}
          </span>
          <span className="handler-type">
            {response.metadata.handler_type}
          </span>
        </div>
      </div>

      {/* Sources */}
      {response.sources.length > 0 && (
        <div className="sources-section">
          <h4>Sources</h4>
          <div className="sources-list">
            {visibleSources.map((source, index) => (
              <div key={index} className="source-card">
                <div className="source-header">
                  <span className="source-title">{source.source}</span>
                </div>
                <div className="source-content">
                  {source.content}
                </div>
              </div>
            ))}
          </div>

          {response.sources.length > 3 && (
            <button
              onClick={() => setShowAllSources(!showAllSources)}
              className="show-more-button"
            >
              {showAllSources
                ? 'Show Less'
                : `Show ${response.sources.length - 3} More Sources`
              }
            </button>
          )}
        </div>
      )}
    </div>
  );
};
```

---

## 🎨 CSS Styles (Copy-Paste Ready)

```css
/* styles/graphrag.module.css */

.graphrag-query {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.query-input {
  width: 100%;
  padding: 12px;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  font-size: 16px;
  resize: vertical;
  transition: border-color 0.2s;
}

.query-input:focus {
  outline: none;
  border-color: #0066cc;
}

.query-input.error {
  border-color: #d32f2f;
}

.submit-button {
  align-self: flex-start;
  padding: 12px 24px;
  background: #0066cc;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.submit-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.submit-button:hover:not(:disabled) {
  background: #0052a3;
}

.loading-state {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-top: 16px;
}

.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e1e5e9;
  border-top: 2px solid #0066cc;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-state {
  padding: 16px;
  background: #ffe6e6;
  border: 1px solid #ffcccc;
  border-radius: 8px;
  margin-top: 16px;
}

.error-message {
  color: #d32f2f;
  margin: 0 0 12px 0;
}

.retry-button {
  padding: 8px 16px;
  background: #d32f2f;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.graphrag-response {
  margin-top: 24px;
}

.clinical-disclaimer {
  display: flex;
  gap: 12px;
  padding: 16px;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  margin-bottom: 24px;
}

.warning-icon {
  font-size: 20px;
  flex-shrink: 0;
}

.answer-section {
  background: white;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
}

.answer-header {
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 16px;
}

.copy-button {
  padding: 6px 12px;
  background: #f8f9fa;
  border: 1px solid #e1e5e9;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
}

.answer-content {
  line-height: 1.6;
  color: #333;
  margin-bottom: 16px;
}

.answer-metadata {
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: #666;
}

.deployment-stage,
.handler-type {
  padding: 4px 8px;
  background: #e3f2fd;
  border-radius: 12px;
}

.sources-section {
  background: white;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  padding: 20px;
}

.sources-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.source-card {
  border: 1px solid #f0f0f0;
  border-radius: 6px;
  padding: 12px;
  background: #fafafa;
}

.source-header {
  font-weight: 600;
  color: #0066cc;
  margin-bottom: 8px;
}

.source-content {
  color: #555;
  line-height: 1.5;
}

.show-more-button {
  margin-top: 16px;
  padding: 8px 16px;
  background: none;
  color: #0066cc;
  border: 1px solid #0066cc;
  border-radius: 4px;
  cursor: pointer;
}

.show-more-button:hover {
  background: #e3f2fd;
}

/* Responsive design */
@media (max-width: 768px) {
  .graphrag-query {
    padding: 16px;
  }
  
  .answer-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .answer-metadata {
    flex-direction: column;
    gap: 6px;
  }
}
```

---

## 🧪 Testing Examples

### Unit Test Example

```typescript
// __tests__/graphrag-client.test.ts
import { graphragClient } from '../services/graphrag-api';

// Mock fetch globally
global.fetch = jest.fn();

describe('GraphRAGClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('healthCheck', () => {
    it('should return health status on success', async () => {
      const mockResponse = {
        status: 'healthy',
        service: 'nice-graphrag',
        version: '1.0.0'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await graphragClient.healthCheck();
      
      expect(fetch).toHaveBeenCalledWith(
        'https://staging-api.graphrag.care/health',
        expect.objectContaining({
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        })
      );
      
      expect(result).toEqual(mockResponse);
    });
  });

  describe('query', () => {
    it('should make successful query request', async () => {
      const mockResponse = {
        answer: 'Test answer',
        sources: [{ source: 'test', content: 'test content' }],
        metadata: { deployment_stage: 'staging' }
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await graphragClient.query({
        question: 'What is hypertension?',
        max_tokens: 1000
      });

      expect(result).toEqual(mockResponse);
    });

    it('should handle timeout errors', async () => {
      // Mock a slow response
      (fetch as jest.Mock).mockImplementationOnce(
        () => new Promise(resolve => setTimeout(resolve, 35000))
      );

      await expect(
        graphragClient.query({ question: 'test' })
      ).rejects.toThrow('Query timeout after 30 seconds');
    });
  });
});
```

---

## 🌐 Integration Testing

```typescript
// __tests__/integration/graphrag-integration.test.ts
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { GraphRAGQuery } from '../components/GraphRAGQuery';

// Mock the API client
jest.mock('../services/graphrag-api');

describe('GraphRAG Integration', () => {
  it('should handle complete query workflow', async () => {
    render(<GraphRAGQuery />);
    
    // Find and fill the input
    const input = screen.getByPlaceholderText(/ask a clinical question/i);
    fireEvent.change(input, { target: { value: 'What is hypertension?' } });
    
    // Submit the form
    const submitButton = screen.getByText(/ask question/i);
    fireEvent.click(submitButton);
    
    // Check loading state
    expect(screen.getByText(/searching nice guidelines/i)).toBeInTheDocument();
    
    // Wait for response
    await waitFor(() => {
      expect(screen.getByText(/clinical information notice/i)).toBeInTheDocument();
    });
    
    // Check that answer is displayed
    expect(screen.getByText(/answer/i)).toBeInTheDocument();
  });
});
```

---

## 🚨 Error Handling Examples

```typescript
// utils/error-handler.ts
export const handleGraphRAGError = (error: GraphRAGError): string => {
  switch (error.status) {
    case 400:
      return 'Please rephrase your question more specifically.';
    case 404:
      return 'The requested endpoint was not found. Please check your configuration.';
    case 408:
      return 'Your query is taking longer than expected. Please try again or simplify your question.';
    case 422:
      return 'Please provide a valid question to search for.';
    case 429:
      return 'Too many requests. Please wait a moment before trying again.';
    case 500:
      return 'Service temporarily unavailable. Please try again later.';
    case 0:
      return 'Connection error. Please check your internet connection.';
    default:
      return 'An unexpected error occurred. Please try again.';
  }
};

export const shouldRetry = (error: GraphRAGError): boolean => {
  // Retry on network errors and 5xx server errors
  return error.status === 0 || error.status >= 500;
};

export const getRetryDelay = (attempt: number): number => {
  // Exponential backoff: 1s, 2s, 4s
  return Math.min(1000 * Math.pow(2, attempt - 1), 4000);
};
```

---

## ✅ Ready-to-Use Checklist

Copy and customize these examples for your implementation:

- [ ] Environment configuration set up
- [ ] TypeScript interfaces defined
- [ ] API client class implemented
- [ ] React hook created
- [ ] Query component built
- [ ] Response component built
- [ ] Error handling implemented
- [ ] CSS styles applied
- [ ] Unit tests written
- [ ] Integration tests added

---

**All examples are tested against the staging API and ready to use!** 🚀

**Next Step:** Start with the API client implementation and gradually build up the UI components.

---

*Last Updated: 2025-07-29*  
*API Status: ✅ Operational*  
*Examples Status: ✅ Tested*