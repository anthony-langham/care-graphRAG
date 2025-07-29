import { SSTConfig } from "sst";
import { Api, Cron, Config } from "sst/constructs";

export default {
  config(_input) {
    return {
      name: "nice-cks-graphrag",
      region: "eu-west-2",
    };
  },
  stacks(app) {
    app.stack(function API({ stack }) {
      // Secrets for secure credential storage
      const MONGODB_URI = new Config.Secret(stack, "MONGODB_URI");
      const OPENAI_API_KEY = new Config.Secret(stack, "OPENAI_API_KEY");

      // Lambda Layer for dependencies - SST manages this automatically
      // Layer files should be in layers/python/ directory

      // API Lambda with optimized settings
      const api = new Api(stack, "api", {
        routes: {
          "POST /query": {
            function: {
              handler: "lambda-functions/query/handler.handler",
              runtime: "python3.11",
              timeout: "30 seconds",
              memorySize: "1024 MB",
              tracing: "active", // Enable X-Ray tracing
              environment: {
                MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
                MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
                MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
                QUERY_TIMEOUT_SECONDS: "25", // 5s buffer for Lambda processing
                MAX_CONTEXT_TOKENS: "2000",
                OPENAI_MODEL: "gpt-4o-mini",
                OPENAI_TEMPERATURE: "0.1",
                LOG_LEVEL: "INFO",
                ENVIRONMENT: "production",
              },
              bind: [MONGODB_URI, OPENAI_API_KEY],
            },
          },
          "GET /health": {
            function: {
              handler: "lambda-functions/health/handler.handler",
              runtime: "python3.11",
              timeout: "15 seconds",
              memorySize: "512 MB",
              tracing: "active", // Enable X-Ray tracing
              environment: {
                MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
                MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
                MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
                LOG_LEVEL: "INFO",
                ENVIRONMENT: "production",
              },
              bind: [MONGODB_URI, OPENAI_API_KEY],
            },
          },
        },
        defaults: {
          function: {
            runtime: "python3.11",
            tracing: "active", // Enable X-Ray tracing for all functions
          },
        },
        cors: {
          allowCredentials: true,
          allowHeaders: ["content-type", "authorization", "x-api-key"],
          allowMethods: ["GET", "POST", "OPTIONS"],
          allowOrigins: [
            "https://care.engineering",
            "https://www.care.engineering",
            process.env.ALLOWED_ORIGIN || "http://localhost:3000", // For development
          ],
        },
      });

      // Note: API Key authentication can be configured manually in AWS Console
      // SST v2 has limited support for complex API Gateway configurations

      // CloudWatch monitoring setup
      // Note: CloudWatch log groups are automatically created by AWS Lambda
      // Manual configuration needed for:
      // 1. Log retention policies (set to 30 days for query, 7 days for health, 90 days for sync)
      // 2. CloudWatch alarms (error rates, duration thresholds)
      // 3. CloudWatch dashboard (Lambda metrics visualization)
      //
      // Recommended manual setup in AWS Console:
      // - Query function error alarm: >5 errors in 5 minutes
      // - Query duration alarm: average >25 seconds
      // - Sync function error alarm: any errors
      // - Dashboard with Lambda metrics: invocations, errors, duration

      // Output the API URL for integration
      stack.addOutputs({
        ApiUrl: api.url,
        CloudWatchDashboard: `https://eu-west-2.console.aws.amazon.com/cloudwatch/home?region=eu-west-2#dashboards:`,
        XRayTraces: `https://eu-west-2.console.aws.amazon.com/xray/home?region=eu-west-2#/traces`,
      });

      // Note: Sync cron function temporarily disabled for initial deployment
      // Will be re-enabled after successful API deployment
      // new Cron(stack, "sync", {
      //   schedule: "rate(7 days)", // Weekly sync as per CLAUDE.md
      //   job: {
      //     function: {
      //       handler: "functions/sync.scheduled_handler",
      //       runtime: "python3.11",
      //       timeout: "5 minutes", // 5 minutes for sync as per CLAUDE.md
      //       memorySize: "2048 MB", // Higher memory for sync operations
      //       tracing: "active", // Enable X-Ray tracing for sync function
      //       install: ["requirements-lambda.txt"],
      //       environment: {
      //         MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
      //         MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
      //         MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
      //         MONGODB_AUDIT_COLLECTION: process.env.MONGODB_AUDIT_COLLECTION || "audit_log",
      //         SYNC_TIMEOUT_SECONDS: "280", // 20s buffer for Lambda processing
      //         BATCH_SIZE: "50", // Process chunks in batches
      //         OPENAI_MODEL: "gpt-4o-mini",
      //         OPENAI_TEMPERATURE: "0.0", // Zero temperature for consistent extraction
      //         LOG_LEVEL: "INFO",
      //         ENVIRONMENT: "production",
      //       },
      //       bind: [MONGODB_URI, OPENAI_API_KEY],
      //     },
      //   },
      // });
    });
  },
} satisfies SSTConfig;
