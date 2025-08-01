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
    const isProduction = $app.stage === "production";
    
    // Create secrets for secure credential storage
    const mongodbUri = new sst.Secret("MongoDbUri");
    const openaiApiKey = new sst.Secret("OpenAiApiKey");
    
    // API Key secret for production authentication
    const apiKey = isProduction ? new sst.Secret("ApiKey") : null;

    // Production-specific CORS settings
    const corsOrigins = isProduction 
      ? [
          "https://care.engineering",
          "https://www.care.engineering"
        ]
      : [
          "https://care.engineering",
          "https://www.care.engineering",
          "http://localhost:3000",
          "http://localhost:5173"
        ];

    // API Gateway with enhanced production settings
    const api = new sst.aws.ApiGatewayV2("Api", {
      cors: {
        allowCredentials: true,
        allowHeaders: ["content-type", "authorization", "x-api-key"],
        allowMethods: ["GET", "POST", "OPTIONS"],
        allowOrigins: corsOrigins,
        maxAge: "86400 seconds", // 24 hours for production
      },
    });

    // Common Lambda settings
    const commonLambdaSettings = {
      runtime: "python3.11" as const,
      tracing: isProduction ? "active" : "pass-through",
      architecture: "arm64" as const, // Cost optimization
      environment: {
        MONGODB_DB_NAME: "ckshtn",
        MONGODB_GRAPH_COLLECTION: "kg",
        MONGODB_VECTOR_COLLECTION: "chunks",
        LOG_LEVEL: isProduction ? "WARNING" : "INFO",
        ENVIRONMENT: $app.stage,
        ENABLE_XRAY: isProduction ? "true" : "false",
      },
    };

    // Query endpoint with production optimizations
    api.route("POST /query", {
      handler: "functions/src/functions/query.handler",
      link: [mongodbUri, openaiApiKey, ...(apiKey ? [apiKey] : [])],
      ...commonLambdaSettings,
      timeout: "30 seconds",
      memory: isProduction ? "2048 MB" : "1024 MB", // More memory for production
      environment: {
        ...commonLambdaSettings.environment,
        QUERY_TIMEOUT_SECONDS: "25",
        MAX_CONTEXT_TOKENS: "2000",
        OPENAI_MODEL: "gpt-4o-mini",
        OPENAI_TEMPERATURE: "0.1",
        RATE_LIMIT_ENABLED: isProduction ? "true" : "false",
        RATE_LIMIT_REQUESTS: "10", // per user per minute
        RATE_LIMIT_WINDOW: "60", // seconds
      },
    });

    // Health endpoint
    api.route("GET /health", {
      handler: "functions/src/functions/health.handler",
      link: [mongodbUri, openaiApiKey],
      ...commonLambdaSettings,
      timeout: "15 seconds",
      memory: "512 MB",
      environment: {
        ...commonLambdaSettings.environment,
        CHECK_DEPENDENCIES: isProduction ? "true" : "false",
      },
    });

    // Production-only: Sync endpoint for automated updates
    if (isProduction) {
      api.route("POST /sync", {
        handler: "functions/src/functions/sync.handler",
        link: [mongodbUri, openaiApiKey, apiKey],
        ...commonLambdaSettings,
        timeout: "5 minutes",
        memory: "3008 MB",
        environment: {
          ...commonLambdaSettings.environment,
          SYNC_ENABLED: "true",
          NICE_BASE_URL: "https://cks.nice.org.uk",
        },
      });
    }

    // Production monitoring: CloudWatch Dashboard
    if (isProduction) {
      // This would typically be defined using CDK constructs
      // For SST v3, we'll create monitoring via AWS console or separate IaC
    }

    // Output the API URL and monitoring links
    return {
      ApiUrl: api.url,
      Stage: $app.stage,
      Region: $app.providers?.aws?.region || "eu-west-2",
      CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:`,
      XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
      LogsInsights: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#logs-insights:`,
      ...(isProduction && {
        SecurityNote: "API Key authentication is enabled. Set x-api-key header in requests.",
        RateLimiting: "10 requests per minute per user",
      }),
    };
  },
});