/**
 * Lambda performance configuration for NICE CKS GraphRAG system.
 * TASK-044: Configure Lambda settings (memory, timeout, concurrency)
 * 
 * This file defines the Lambda performance settings for different function types
 * and provides guidance for optimization based on CloudWatch metrics.
 */

export interface LambdaPerformanceConfig {
  memory: number;          // Memory allocation in MB
  timeout: number;         // Timeout in seconds
  concurrency: number;     // Reserved concurrent executions
  description: string;     // Configuration explanation
  optimization_notes: string[]; // Performance tuning guidance
}

/**
 * Lambda performance configurations for different function types.
 * Based on CLAUDE.md requirements and AWS Lambda best practices.
 */
export const LAMBDA_PERFORMANCE_CONFIGS: Record<string, LambdaPerformanceConfig> = {
  query: {
    memory: 1024,
    timeout: 30,
    concurrency: 20,
    description: "Main QA endpoint with hybrid retrieval and GPT-4o-mini processing",
    optimization_notes: [
      "Memory: Start with 1024MB, monitor CloudWatch metrics",
      "Timeout: 30s allows 25s processing + 5s buffer",
      "Concurrency: 20 concurrent queries for cost control",
      "Monitor: Duration, memory utilization, error rate",
      "Scale up: Increase memory if duration consistently > 20s",
      "Scale down: Reduce memory if utilization < 50%"
    ]
  },
  
  health: {
    memory: 512,
    timeout: 15,
    concurrency: 5,
    description: "Health check endpoint with minimal processing requirements",
    optimization_notes: [
      "Memory: 512MB sufficient for health checks",
      "Timeout: 15s adequate for MongoDB connection test",
      "Concurrency: 5 concurrent health checks sufficient",
      "Monitor: Success rate, cold start frequency",
      "Keep lightweight: No heavy processing or large dependencies"
    ]
  },
  
  sync: {
    memory: 2048,
    timeout: 300,
    concurrency: 1,
    description: "Scheduled sync operations with web scraping and graph building",
    optimization_notes: [
      "Memory: 2048MB for processing large documents and graph operations",
      "Timeout: 5 minutes (300s) allows 280s processing + 20s buffer",
      "Concurrency: 1 ensures only one sync runs at a time",
      "Monitor: Duration, memory peaks, batch processing efficiency",
      "Consider: Step Functions if sync exceeds 15 minutes",
      "Batch processing: Process documents in chunks of 50"
    ]
  }
};

/**
 * Cost optimization guidelines based on Lambda pricing model.
 * EU-West-2 pricing (as of 2024): $0.0000166667 per GB-second + $0.0000002 per request
 */
export const COST_OPTIMIZATION_GUIDELINES = {
  memory_vs_cost: {
    description: "Higher memory = faster execution but higher cost per second",
    recommendations: [
      "Measure actual memory usage via CloudWatch",
      "Right-size memory to avoid over-provisioning",
      "Consider execution time vs memory cost trade-off",
      "Test memory configurations: 512MB, 1024MB, 1536MB, 2048MB"
    ]
  },
  
  concurrency_limits: {
    description: "Reserved concurrency prevents runaway costs",
    recommendations: [
      "Query endpoint: 20 concurrent executions max",
      "Health endpoint: 5 concurrent executions max", 
      "Sync endpoint: 1 concurrent execution (prevents overlaps)",
      "Monitor throttling metrics to adjust limits"
    ]
  },
  
  cold_start_optimization: {
    description: "Minimize cold start impact on performance",
    recommendations: [
      "Use Lambda layers for dependencies",
      "Implement connection pooling (MongoDB)",
      "Cache configuration and secrets",
      "Consider provisioned concurrency for critical functions"
    ]
  }
};

/**
 * CloudWatch metrics to monitor for performance optimization.
 */
export const MONITORING_METRICS = {
  primary_metrics: [
    "Duration",           // Execution time
    "Memory",            // Memory utilization 
    "ConcurrentExecutions", // Active concurrent executions
    "Errors",            // Error count
    "Throttles",         // Throttling events
    "IteratorAge"        // For event-driven functions
  ],
  
  custom_metrics: [
    "MongoDB_Connection_Time", // Custom metric for DB connection latency
    "OpenAI_API_Latency",     // Custom metric for LLM API calls
    "Graph_Traversal_Time",   // Custom metric for graph operations
    "Vector_Search_Time",     // Custom metric for vector operations
    "Cost_Per_Query"          // Custom metric for cost tracking
  ],
  
  alarms: [
    {
      metric: "Duration",
      threshold: "25000", // 25 seconds
      description: "Alert if query processing approaches timeout"
    },
    {
      metric: "Memory",
      threshold: "90", // 90% utilization
      description: "Alert if memory utilization too high"
    },
    {
      metric: "Errors",
      threshold: "5", // 5 errors in 5 minutes
      description: "Alert on error rate increase"
    },
    {
      metric: "Throttles", 
      threshold: "1", // Any throttling
      description: "Alert on concurrency throttling"
    }
  ]
};

/**
 * Performance testing recommendations for Lambda functions.
 */
export const PERFORMANCE_TESTING = {
  load_testing: {
    query_endpoint: {
      concurrent_users: [1, 5, 10, 20, 25], // Test up to concurrency limit
      test_duration: "10 minutes",
      ramp_up_time: "2 minutes",
      expected_response_time: "< 20 seconds",
      success_rate: "> 95%"
    }
  },
  
  memory_profiling: {
    test_configurations: [512, 1024, 1536, 2048, 3008], // MB
    metrics_to_track: ["Duration", "Cost", "Memory_Utilization"],
    test_scenarios: [
      "Simple medical questions",
      "Complex multi-part queries", 
      "Questions requiring vector fallback",
      "Edge cases and error conditions"
    ]
  },
  
  timeout_testing: {
    scenarios: [
      "Normal queries (< 15s expected)",
      "Complex queries (15-25s expected)", 
      "Timeout edge cases (25-30s)",
      "MongoDB connection issues",
      "OpenAI API latency spikes"
    ]
  }
};

/**
 * Environment-specific configurations.
 */
export const ENVIRONMENT_CONFIGS = {
  development: {
    memory_multiplier: 0.5, // Use less memory in dev
    timeout_buffer: 10, // Longer timeout buffer for debugging
    concurrency_limit: 2, // Lower concurrency in dev
    logging_level: "DEBUG"
  },
  
  staging: {
    memory_multiplier: 0.8, // Slightly less memory than prod
    timeout_buffer: 7, // Standard timeout buffer
    concurrency_limit: 10, // Half of production concurrency
    logging_level: "INFO"
  },
  
  production: {
    memory_multiplier: 1.0, // Full memory allocation
    timeout_buffer: 5, // Minimal timeout buffer
    concurrency_limit: 20, // Full concurrency
    logging_level: "INFO"
  }
};

export default {
  LAMBDA_PERFORMANCE_CONFIGS,
  COST_OPTIMIZATION_GUIDELINES,
  MONITORING_METRICS,
  PERFORMANCE_TESTING,
  ENVIRONMENT_CONFIGS
};