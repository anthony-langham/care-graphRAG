# GraphRAG Specialized Agents

This directory contains 16 specialized subagents designed for the Care-GraphRAG system. Each agent is an expert in a specific domain, providing focused assistance for medical AI development, deployment, and maintenance.

## Agent Categories

### 🏥 Medical/Clinical Domain Agents
Expert agents for medical content validation and clinical safety:

- **[clinical-validator](./clinical-validator.json)** - Validates medical content against NICE guidelines and clinical standards
- **[medical-entity-extractor](./medical-entity-extractor.json)** - Extracts clinical entities, medications, and treatment pathways
- **[guideline-analyzer](./guideline-analyzer.json)** - Analyzes clinical guidelines for consistency and completeness

### 🕸️ GraphRAG-Specific Agents  
Specialized agents for knowledge graph and retrieval optimization:

- **[graph-optimizer](./graph-optimizer.json)** - Optimizes graph structure, indexes, and query performance
- **[retrieval-tuner](./retrieval-tuner.json)** - Fine-tunes hybrid retrieval parameters and fallback strategies
- **[knowledge-graph-builder](./knowledge-graph-builder.json)** - Constructs and maintains medical knowledge graphs
- **[vector-store-manager](./vector-store-manager.json)** - Manages embeddings, indexes, and vector search optimization

### 📊 Data & Quality Agents
Agents focused on data quality, testing, and performance analysis:

- **[medical-scraper](./medical-scraper.json)** - Specialized web scraping for medical/clinical content
- **[data-validator](./data-validator.json)** - Validates data quality, completeness, and medical accuracy
- **[test-generator](./test-generator.json)** - Generates comprehensive test cases for medical AI systems
- **[performance-analyzer](./performance-analyzer.json)** - Analyzes system performance, costs, and optimization opportunities

### ⚙️ Infrastructure & DevOps Agents
Infrastructure optimization and operations specialists:

- **[aws-lambda-optimizer](./aws-lambda-optimizer.json)** - Optimizes Lambda functions, layers, and serverless architecture
- **[mongodb-specialist](./mongodb-specialist.json)** - MongoDB schema design, indexing, and performance tuning
- **[security-auditor](./security-auditor.json)** - Security reviews for healthcare/clinical systems
- **[cost-optimizer](./cost-optimizer.json)** - Analyzes and optimizes cloud costs for AI systems

### 🛠️ Development Agents
Software development and deployment specialists:

- **[api-designer](./api-designer.json)** - Designs REST APIs and integration patterns
- **[documentation-writer](./documentation-writer.json)** - Creates technical documentation and API specs
- **[code-reviewer](./code-reviewer.json)** - Reviews code for medical AI systems with clinical safety focus
- **[deployment-manager](./deployment-manager.json)** - Manages CI/CD pipelines and production deployments

## Usage Patterns

### Proactive Agents
These agents automatically engage when relevant conditions are detected:
- `clinical-validator` - Validates medical content after extraction
- `data-validator` - Checks data quality during pipeline operations
- `graph-optimizer` - Optimizes performance when queries are slow
- `retrieval-tuner` - Tunes retrieval parameters for better accuracy
- `performance-analyzer` - Monitors system performance continuously
- `security-auditor` - Reviews security configurations proactively
- `cost-optimizer` - Monitors and optimizes costs automatically
- `code-reviewer` - Reviews code after significant implementations
- `deployment-manager` - Manages deployment processes

### On-Demand Agents
These agents are invoked for specific tasks:
- `medical-entity-extractor` - Called for entity extraction tasks
- `guideline-analyzer` - Used for guideline analysis projects
- `knowledge-graph-builder` - Engaged for graph construction
- `vector-store-manager` - Used for vector store configuration
- `medical-scraper` - Called for content scraping tasks
- `test-generator` - Used for test case generation
- `api-designer` - Engaged for API design tasks
- `documentation-writer` - Called for documentation creation

## Integration Examples

### Clinical Safety Workflow
```
User Request → clinical-validator (proactive) → medical-entity-extractor → data-validator (proactive) → graph-optimizer (proactive)
```

### Performance Optimization
```
Slow Queries → performance-analyzer (proactive) → graph-optimizer → retrieval-tuner → cost-optimizer (proactive)
```

### Deployment Pipeline
```
Code Changes → code-reviewer (proactive) → test-generator → security-auditor (proactive) → deployment-manager (proactive)
```

## Agent Configuration

Each agent is defined with:
- **Name**: Unique identifier
- **Description**: Role and specialization
- **Expertise**: List of knowledge domains
- **Tools**: Available tools (all agents have access to "*")
- **Use Cases**: Specific scenarios where the agent helps
- **Prompt Template**: Specialized instructions for the agent
- **Proactive**: Whether the agent engages automatically
- **When to Use**: Specific triggers for agent activation

## Best Practices

1. **Use Multiple Agents**: Complex tasks often benefit from multiple specialized agents
2. **Follow Agent Chains**: Let proactive agents engage automatically in workflows
3. **Leverage Expertise**: Each agent has deep domain knowledge - use their specialized prompts
4. **Clinical Safety First**: Medical agents prioritize patient safety and clinical accuracy
5. **Performance Focus**: Infrastructure agents balance performance, cost, and reliability

## Future Extensions

The agent system is designed to be extensible. New agents can be added for:
- Additional medical specialties
- New infrastructure components
- Emerging AI/ML techniques
- Compliance requirements
- Integration patterns

Each new agent follows the same JSON schema and integration patterns established by these 16 core agents.