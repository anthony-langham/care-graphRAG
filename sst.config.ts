/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "nice-cks-graphrag",
      removal: input?.stage === "production" ? "retain" : "remove",
      home: "aws",
      providers: {
        aws: {
          region: "eu-west-2",
        },
      },
    };
  },
  async run() {
    // Create secrets for secure credential storage
    const mongodbUri = new sst.Secret("MongoDbUri");
    const openaiApiKey = new sst.Secret("OpenAiApiKey");

    // API with Python Lambda functions
    const api = new sst.aws.ApiGatewayV2("Api", {
      cors: {
        allowCredentials: true,
        allowHeaders: ["content-type", "authorization", "x-api-key"],
        allowMethods: ["GET", "POST", "OPTIONS"],
        allowOrigins: [
          "https://care.engineering",
          "https://www.care.engineering",
          process.env.ALLOWED_ORIGIN || "http://localhost:3000",
        ],
      },
    });

    // API Key secret for production authentication
    const apiKey = $app.stage === "production" ? new sst.Secret("ApiKey") : null;

    // Query endpoint
    api.route("POST /query", {
      handler: $app.stage === "production" 
        ? "functions/src/functions/query_prod.handler"
        : "functions/src/functions/query.handler",
      link: [mongodbUri, openaiApiKey, ...(apiKey ? [apiKey] : [])],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: $app.stage === "production" ? "2048 MB" : "1024 MB",
      environment: {
        // MongoDB Configuration
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
        MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
        MONGODB_AUDIT_COLLECTION: process.env.MONGODB_AUDIT_COLLECTION || "audit_log",
        MONGODB_TIMEOUT_MS: "5000",
        
        // OpenAI Configuration
        OPENAI_MODEL: "gpt-4o-mini",
        OPENAI_TEMPERATURE: "0.0",
        
        // Query & Performance Configuration
        QUERY_TIMEOUT_SECONDS: "25",
        MAX_CONTEXT_TOKENS: "2000",
        MAX_RESULTS: "10",
        SIMILARITY_THRESHOLD: "0.7",
        MAX_DEPTH: "3",
        VECTOR_WEIGHT: "0.3",
        
        // Application Configuration
        LOG_LEVEL: $app.stage === "production" ? "WARNING" : "INFO",
        ENVIRONMENT: $app.stage,
        
        // Authentication & Rate Limiting
        RATE_LIMIT_ENABLED: $app.stage === "production" ? "true" : "false",
        RATE_LIMIT_REQUESTS: "10",
        RATE_LIMIT_WINDOW: "60",
        API_KEY: $app.stage === "production" ? "222bfcbcf7d6875344681c6f2fcac133b6907fb2aa5ba7b71a54d603b11b10fc" : "",
      },
    });

    // Health endpoint
    api.route("GET /health", {
      handler: "functions/src/functions/health.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "15 seconds",
      memory: "512 MB",
      environment: {
        // MongoDB Configuration
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
        MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
        MONGODB_AUDIT_COLLECTION: process.env.MONGODB_AUDIT_COLLECTION || "audit_log",
        MONGODB_TIMEOUT_MS: "5000",
        
        // OpenAI Configuration
        OPENAI_MODEL: "gpt-4o-mini",
        OPENAI_TEMPERATURE: "0.0",
        
        // Application Configuration
        LOG_LEVEL: $app.stage === "production" ? "WARNING" : "INFO",
        ENVIRONMENT: $app.stage,
        CHECK_DEPENDENCIES: $app.stage === "production" ? "true" : "false",
      },
    });

    // Environment test endpoint (temporary for debugging)
    api.route("GET /env-test", {
      handler: "functions/src/functions/env_test.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: "1024 MB",
      environment: {
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        ENVIRONMENT: $app.stage,
      },
    });

    // Output the API URL and monitoring links
    return {
      ApiUrl: api.url,
      CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:`,
      XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
    };
  },
});