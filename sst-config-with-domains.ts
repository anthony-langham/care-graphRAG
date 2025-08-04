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
    const alertsTopic = $app.stage === "production" ? new sst.aws.SnsTopic("AlertsTopic") : null;

    // Custom domain configuration based on stage
    let domainProps = {};
    
    if ($app.stage === "production") {
      // For production, we need to create the certificate in us-east-1
      const cert = new sst.aws.AcmCertificate("ApiCertificate", {
        domainName: "api.graphrag.care",
        validation: {
          method: "DNS",
          // You'll need to add these DNS records to Cloudflare
          // The records will be shown in the SST output
        },
      });
      
      domainProps = {
        domain: {
          name: "api.graphrag.care",
          cert: cert.arn,
          dns: false, // We're using Cloudflare for DNS
        },
      };
    } else if ($app.stage === "staging") {
      // For staging
      const cert = new sst.aws.AcmCertificate("ApiCertificate", {
        domainName: "staging-api.graphrag.care",
        validation: {
          method: "DNS",
        },
      });
      
      domainProps = {
        domain: {
          name: "staging-api.graphrag.care",
          cert: cert.arn,
          dns: false, // We're using Cloudflare for DNS
        },
      };
    }

    // API with Python Lambda functions and custom domain
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
      ...domainProps,
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
    const envTestFunction = api.route("GET /env-test", {
      handler: "functions/src/functions/env_test.handler",
      link: [mongodbUri, openaiApiKey],
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

    // Output the API URL and monitoring links
    const outputs: Record<string, any> = {
      ApiUrl: api.url,
      CustomDomain: domainProps.domain?.name || "No custom domain configured",
      CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:name/nice-cks-graphrag-${$app.stage}`,
      XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
    };

    if ($app.stage === "production" && alertsTopic) {
      outputs.AlertsTopicArn = alertsTopic.arn;
    }

    // Show certificate validation instructions if using custom domain
    if (domainProps.domain) {
      console.log("\n⚠️  IMPORTANT: Certificate Validation Required!");
      console.log("1. Check AWS ACM Console for validation DNS records");
      console.log("2. Add the CNAME records to Cloudflare");
      console.log("3. Wait for certificate validation (5-30 minutes)");
      console.log("4. Then redeploy with: sst deploy --stage " + $app.stage);
    }

    return outputs;
  },
});