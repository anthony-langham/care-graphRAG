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

    // Query endpoint
    api.route("POST /query", {
      handler: "functions/src/functions/query.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: "1024 MB",
      environment: {
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
        MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
        QUERY_TIMEOUT_SECONDS: "25",
        MAX_CONTEXT_TOKENS: "2000",
        OPENAI_MODEL: "gpt-4o-mini",
        OPENAI_TEMPERATURE: "0.1",
        LOG_LEVEL: "INFO",
        ENVIRONMENT: $app.stage,
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
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
        MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
        LOG_LEVEL: "INFO",
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