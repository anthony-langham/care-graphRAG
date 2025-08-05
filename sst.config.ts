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

    // Create SNS topic for alerts (production only)
    // Disabled due to IAM permissions - enable after adding SNS permissions to deploy user
    const alertsTopic = null; // $app.stage === "production" ? new sst.aws.SnsTopic("AlertsTopic") : null;


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
       // Add custom domain configuration (temporarily disabled for production update)
       // ...($app.stage === "production" && {
       //   domain: {
       //     name: "api.graphrag.care",
       //     cert: "arn:aws:acm:eu-west-2:146409062658:certificate/04ea5541-2506-4169-880e-f3c60457a82f",
       //     dns: false, // We're managing DNS in Cloudflare
       //   }
       // }),
       ...($app.stage === "staging" && {
         domain: {
           name: "staging-api.graphrag.care",
           cert: "arn:aws:acm:eu-west-2:146409062658:certificate/ee003893-b55d-445d-9981-260fbbfe3aa2",
           dns: false, // We're managing DNS in Cloudflare
         }
       }),
       permissions: [
         {
           actions: ["ssm:GetParameter"],
           resources: [
             `arn:aws:ssm:eu-west-2:146409062658:parameter/nice-cks-graphrag/*`,
           ],
         },
       ],
     });
 
     // API Key secret for production authentication
     const apiKey = $app.stage === "production" ? new sst.Secret("ApiKey") : null;
 
     // Query endpoint with enhanced monitoring
     api.route("POST /query", {
       runtime: "python3.11",
       handler: "functions/src/functions/query_prod.handler",
       timeout: "30 seconds",
       memory: $app.stage === "production" ? "2048 MB" : "1024 MB",
       transform: {
         function: (args) => {
           if ($app.stage === "production") {
             args.tracingConfig = { mode: "Active" };
           }
           args.loggingConfig = {
             logFormat: "JSON",
             logGroup: `/aws/lambda/nice-cks-graphrag-${$app.stage}-query`,
             applicationLogLevel: $app.stage === "production" ? "WARN" : "INFO",
             systemLogLevel: "INFO"
           };
         }
       },
       environment: {
         MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
         MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
         MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
         MONGODB_AUDIT_COLLECTION: process.env.MONGODB_AUDIT_COLLECTION || "audit_log",
         MONGODB_TIMEOUT_MS: "5000",
         OPENAI_MODEL: "gpt-4o-mini",
         OPENAI_TEMPERATURE: "0.0",
         QUERY_TIMEOUT_SECONDS: "25",
         MAX_CONTEXT_TOKENS: "2000",
         MAX_RESULTS: "10",
         SIMILARITY_THRESHOLD: "0.7",
         MAX_DEPTH: "3",
         VECTOR_WEIGHT: "0.3",
         LOG_LEVEL: $app.stage === "production" ? "WARNING" : "INFO",
         ENVIRONMENT: $app.stage,
         RATE_LIMIT_ENABLED: $app.stage === "production" ? "true" : "false",
         RATE_LIMIT_REQUESTS: "10",
         RATE_LIMIT_WINDOW: "60",
       },
     });
 
     // Debug endpoint (temporary)
     api.route("GET /debug/env", {
       handler: "functions/src/functions/debug_env.handler",
       runtime: "python3.11",
       timeout: "10 seconds",
       memory: "512 MB",
     });
 
     // Health endpoint with monitoring
     const healthFunction = api.route("GET /health", {
       handler: "functions/health.handler",
       runtime: "python3.11",
       timeout: "15 seconds",
       memory: "512 MB",
       transform: {
         function: (args) => {
           if ($app.stage === "production") {
             args.tracingConfig = { mode: "Active" };
           }
           args.loggingConfig = {
             logFormat: "JSON",
             logGroup: `/aws/lambda/nice-cks-graphrag-${$app.stage}-health`,
             applicationLogLevel: $app.stage === "production" ? "WARN" : "INFO",
             systemLogLevel: "INFO"
           };
         }
       },
       environment: {
         MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
         MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
         MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
         MONGODB_AUDIT_COLLECTION: process.env.MONGODB_AUDIT_COLLECTION || "audit_log",
         MONGODB_TIMEOUT_MS: "5000",
         OPENAI_MODEL: "gpt-4o-mini",
         OPENAI_TEMPERATURE: "0.0",
         LOG_LEVEL: $app.stage === "production" ? "WARNING" : "INFO",
         ENVIRONMENT: $app.stage,
         CHECK_DEPENDENCIES: $app.stage === "production" ? "true" : "false",
       },
     });
 
     // Environment test endpoint (temporary for debugging)
     const envTestFunction = api.route("GET /env-test", {
       handler: "functions/src/functions/env_test.handler",
       runtime: "python3.11",
       timeout: "30 seconds",
       memory: "1024 MB",
       transform: {
         function: (args) => {
           // Enable X-Ray tracing for production
           if ($app.stage === "production") {
             args.tracingConfig = { mode: "Active" };
           }
         }
       },
       environment: {
         MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
         ENVIRONMENT: $app.stage,
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

    // Custom domain info for outputs
    const customDomain = $app.stage === "production" 
      ? "api.graphrag.care"
      : $app.stage === "staging" 
      ? "staging-api.graphrag.care"
      : undefined;

    // Output the API URL and monitoring links
    const outputs: Record<string, any> = {
      ApiUrl: api.url,
      CustomDomain: customDomain || "No custom domain configured",
      CustomDomainStatus: "Pending certificate setup - run ./scripts/setup-api-gateway-domains.sh",
      CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:name/nice-cks-graphrag-${$app.stage}`,
      XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
    };

    // Commented out until SNS permissions are added to deploy user
    // if ($app.stage === "production" && alertsTopic) {
    //   outputs.AlertsTopicArn = alertsTopic.arn;
    // }

    return outputs;
  },
});
