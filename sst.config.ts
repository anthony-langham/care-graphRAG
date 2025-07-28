import { SSTConfig } from "sst";
import { Api, LayerVersion, Cron, Config } from "sst/constructs";
import { Code, Runtime } from "aws-cdk-lib/aws-lambda";
import { ApiKey, UsagePlan, Period, MethodOptions } from "aws-cdk-lib/aws-apigateway";

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
      const MONGODB_URI = new Config.Secret(stack, "MONGODB_URI", {
        description: "MongoDB Atlas connection string for NICE CKS GraphRAG",
      });
      
      const OPENAI_API_KEY = new Config.Secret(stack, "OPENAI_API_KEY", {
        description: "OpenAI API key for GPT-4o-mini model access",
      });

      // Lambda Layer for dependencies
      const layer = new LayerVersion(stack, "PythonDeps", {
        code: Code.fromAsset("layers/python"),
        compatibleRuntimes: [Runtime.PYTHON_3_11],
        description: "Python dependencies for NICE CKS GraphRAG Lambda functions",
      });

      // API Lambda with optimized settings
      const api = new Api(stack, "api", {
        routes: {
          "POST /query": "functions/query.handler",
          "GET /health": "functions/health.handler", // Health check doesn't need auth
        },
        defaults: {
          function: {
            runtime: "python3.11",
            layers: [layer],
            timeout: 30, // 30s for queries as per CLAUDE.md
            memorySize: 1024, // Start with 1024MB, adjust based on CloudWatch metrics
            environment: {
              MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
              MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
              MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
              // Lambda-specific optimizations
              PYTHONPATH: "/opt/python:/var/task",
            },
            bind: [MONGODB_URI, OPENAI_API_KEY],
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

      // API Key authentication for production security
      const apiKey = new ApiKey(stack, "ApiKey", {
        apiKeyName: "care-engineering-api-key",
        description: "API key for care.engineering to access NICE CKS GraphRAG",
      });

      // Usage plan with rate limiting for cost control
      const usagePlan = new UsagePlan(stack, "UsagePlan", {
        name: "care-engineering-usage-plan",
        description: "Usage plan for care.engineering NICE CKS GraphRAG access",
        throttle: {
          rateLimit: 10, // 10 requests per second
          burstLimit: 20, // 20 request burst capacity
        },
        quota: {
          limit: 10000, // 10,000 requests per day
          period: Period.DAY,
        },
      });

      // Associate the API key with the usage plan
      usagePlan.addApiKey(apiKey);

      // Add API stages to the usage plan and configure API key auth
      if (api.cdk && api.cdk.restApi) {
        usagePlan.addApiStage({
          api: api.cdk.restApi,
          stage: api.cdk.restApi.deploymentStage,
        });

        // Configure API key requirement for /query endpoint
        const queryResource = api.cdk.restApi.root.getResource("query");
        if (queryResource) {
          const postMethod = queryResource.getMethod("POST");
          if (postMethod) {
            // Update the method to require API key
            const cfnMethod = postMethod.node.defaultChild as any;
            cfnMethod.apiKeyRequired = true;
          }
        }
      }

      // Output the API key ID for retrieval
      stack.addOutputs({
        ApiUrl: api.url,
        ApiKeyId: apiKey.keyId,
      });

      // Scheduled sync
      new Cron(stack, "sync", {
        schedule: "rate(7 days)",
        job: {
          function: {
            handler: "functions/sync.scheduled_handler",
            runtime: "python3.11",
            layers: [layer],
            timeout: 300, // 5 minutes for sync as per CLAUDE.md
            memorySize: 1024,
            environment: {
              MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
              MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
              MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
              PYTHONPATH: "/opt/python:/var/task",
            },
            bind: [MONGODB_URI, OPENAI_API_KEY],
          },
        },
      });
    });
  },
} satisfies SSTConfig;
