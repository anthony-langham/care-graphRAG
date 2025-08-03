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
        cloudflare: {
          apiToken: process.env.CLOUDFLARE_API_TOKEN,
        },
      },
    };
  },
  async run() {
    // Create secrets for secure credential storage
    const mongodbUri = new sst.Secret("MongoDbUri");
    const openaiApiKey = new sst.Secret("OpenAiApiKey");

    // Create SNS topic for alerts (production only)
    const alertsTopic = $app.stage === "production" ? new sst.aws.SnsTopic("AlertsTopic") : null;

    // Custom domain configuration
    const domainName = $app.stage === "production" 
      ? "api.graphrag.care" 
      : "staging-api.graphrag.care";

    // API with custom domain
    const api = new sst.aws.ApiGatewayV2("Api", {
      cors: {
        allowCredentials: true,
        allowHeaders: ["content-type", "authorization", "x-api-key"],
        allowMethods: ["GET", "POST", "OPTIONS"],
        allowOrigins: [
          "https://care.engineering",
          "https://www.care.engineering",
          "https://graphrag.care",
          "https://www.graphrag.care",
          process.env.ALLOWED_ORIGIN || "http://localhost:3000",
        ],
      },
      domain: {
        name: domainName,
        dns: sst.cloudflare.dns(),
      },
    });

    // API Key secret for production authentication
    const apiKey = $app.stage === "production" ? new sst.Secret("ApiKey") : null;

    // Query endpoint with enhanced monitoring
    const queryFunction = api.route("POST /query", {
      handler: "functions/src/functions/query_prod.handler",
      link: [mongodbUri, openaiApiKey, ...(apiKey ? [apiKey] : [])],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: $app.stage === "production" ? "2048 MB" : "1024 MB",
      transform: {
        function: (args) => {
          // Enable X-Ray tracing for production
          if ($app.stage === "production") {
            args.tracingConfig = { mode: "Active" };
          }
          // Add enhanced CloudWatch insights
          args.loggingConfig = {
            logFormat: "JSON",
            logGroup: `/aws/lambda/nice-cks-graphrag-${$app.stage}-query`,
            applicationLogLevel: $app.stage === "production" ? "WARN" : "INFO",
            systemLogLevel: "INFO"
          };
        }
      },
      environment: {
        // Domain configuration
        API_DOMAIN: domainName,
        ALLOWED_ORIGINS: JSON.stringify([
          "https://care.engineering",
          "https://www.care.engineering", 
          "https://graphrag.care",
          "https://www.graphrag.care"
        ]),
        
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
      },
    });

    // Health endpoint with monitoring
    const healthFunction = api.route("GET /health", {
      handler: "functions/src/functions/health.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "15 seconds",
      memory: "512 MB",
      transform: {
        function: (args) => {
          // Enable X-Ray tracing for production
          if ($app.stage === "production") {
            args.tracingConfig = { mode: "Active" };
          }
          // Add enhanced CloudWatch insights
          args.loggingConfig = {
            logFormat: "JSON",
            logGroup: `/aws/lambda/nice-cks-graphrag-${$app.stage}-health`,
            applicationLogLevel: $app.stage === "production" ? "WARN" : "INFO",
            systemLogLevel: "INFO"
          };
        }
      },
      environment: {
        // Domain configuration
        API_DOMAIN: domainName,
        
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

    // CloudWatch alarms for production monitoring (using Pulumi directly)
    if ($app.stage === "production" && alertsTopic) {
      // We'll use a simpler approach and create alarms via AWS CDK construct after deployment
      // This avoids SST v3 API compatibility issues while still providing monitoring
      console.log("Production monitoring configured - alarms will be created via setup script");
    }

    // CloudWatch Dashboard will be created via setup script
    // This avoids SST v3 API compatibility issues

    // Output the API URL and monitoring links
    const outputs = {
      ApiUrl: `https://${domainName}`,
      CustomDomain: domainName,
      HealthEndpoint: `https://${domainName}/health`,
      QueryEndpoint: `https://${domainName}/query`,
      CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:name/nice-cks-graphrag-${$app.stage}`,
      XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
    };

    if ($app.stage === "production" && alertsTopic) {
      outputs.AlertsTopicArn = alertsTopic.arn;
    }

    return outputs;
  },
});