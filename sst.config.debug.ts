/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "nice-cks-graphrag",
      removal: "remove",  // Always remove - only using staging
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
          "https://graphrag.care",
          "https://www.graphrag.care",
          process.env.ALLOWED_ORIGIN || "http://localhost:3000",
        ],
      },
      // Custom domain for staging only (temporarily disabled to avoid conflicts)
      // domain: {
      //   name: "staging-api.graphrag.care",
      //   cert: "arn:aws:acm:eu-west-2:146409062658:certificate/ee003893-b55d-445d-9981-260fbbfe3aa2",
      //   dns: false, // We're managing DNS in Cloudflare
      // },
    });


    // Basic test endpoint (no GraphRAG imports)
    api.route("GET /test-basic", {
      handler: "functions/handlers/test_basic.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "10 seconds",
      memory: "512 MB",
      environment: {
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
      },
    });

    // Minimal test endpoint  
    api.route("GET /test-minimal", {
      handler: "functions/handlers/test_minimal.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "10 seconds",
      memory: "512 MB",
      environment: {
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
      },
    });

    // Test imports endpoint
    api.route("GET /test-imports", {
      handler: "functions/handlers/query_prod.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "10 seconds",
      memory: "512 MB",
    });

    // Test QA initialization endpoint
    api.route("GET /test-qa-init", {
      handler: "functions/handlers/query_prod.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "15 seconds",
      memory: "512 MB",
      environment: {
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
        MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
      },
    });

    // Query endpoint with enhanced monitoring
    const queryFunction = api.route("POST /query", {
      handler: "functions/handlers/query_prod.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: "1024 MB",
      transform: {
        function: (args) => {
          // Enable X-Ray tracing
          args.tracingConfig = { mode: "Active" };
          // Add enhanced CloudWatch insights
          args.loggingConfig = {
            logFormat: "JSON",
            logGroup: `/aws/lambda/nice-cks-graphrag-${$app.stage}-query`,
            applicationLogLevel: "INFO",
            systemLogLevel: "INFO"
          };
        }
      },
      environment: {
        // Pass secrets directly
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
        
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
        LOG_LEVEL: "INFO",
        ENVIRONMENT: $app.stage,
        
        // Authentication & Rate Limiting
        RATE_LIMIT_ENABLED: "true",
        RATE_LIMIT_REQUESTS: "10",
        RATE_LIMIT_WINDOW: "60",
      },
    });

    // Debug endpoint (temporary)
    api.route("GET /debug/env", {
      handler: "functions/handlers/debug_env.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "10 seconds",
      memory: "512 MB",
    });

    // Health endpoint with monitoring
    const healthFunction = api.route("GET /health", {
      handler: "functions/handlers/health.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "15 seconds",
      memory: "512 MB",
      transform: {
        function: (args) => {
          // Enable X-Ray tracing
          args.tracingConfig = { mode: "Active" };
          // Add enhanced CloudWatch insights
          args.loggingConfig = {
            logFormat: "JSON",
            logGroup: `/aws/lambda/nice-cks-graphrag-${$app.stage}-health`,
            applicationLogLevel: "INFO",
            systemLogLevel: "INFO"
          };
        }
      },
      environment: {
        // Pass secrets directly
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
        
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
        LOG_LEVEL: "INFO",
        ENVIRONMENT: $app.stage,
        CHECK_DEPENDENCIES: "true",
      },
    });

    // Secrets debugging endpoint 
    const secretsDebugFunction = api.route("GET /debug-secrets", {
      handler: "functions/handlers/debug_secrets.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: "1024 MB",
      transform: {
        function: (args) => {
          // Enable X-Ray tracing
          args.tracingConfig = { mode: "Active" };
        }
      },
      environment: {
        // Pass secrets directly for comparison
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
        
        // Configuration
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        ENVIRONMENT: $app.stage,
      },
    });

    // MongoDB test endpoint (debugging)
    const mongoTestFunction = api.route("GET /test-mongodb", {
      handler: "functions/handlers/test_mongodb.handler",
      link: [mongodbUri, openaiApiKey],
      runtime: "python3.11",
      timeout: "30 seconds",
      memory: "1024 MB",
      transform: {
        function: (args) => {
          // Enable X-Ray tracing
          args.tracingConfig = { mode: "Active" };
        }
      },
      environment: {
        // Pass secrets directly
        MONGODB_URI: process.env.MONGODB_URI || "",
        OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
        
        // MongoDB Configuration
        MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
        ENVIRONMENT: $app.stage,
      },
    });


    // Output the API URL and monitoring links
    return {
      ApiUrl: api.url,
      CustomDomain: "staging-api.graphrag.care",
      CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:name/nice-cks-graphrag-staging`,
      XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
    };
  },
});