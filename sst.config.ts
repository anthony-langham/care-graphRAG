import { SSTConfig } from "sst";
import { Api, LayerVersion, Cron } from "sst/constructs";
import { Code, Runtime } from "aws-cdk-lib/aws-lambda";

export default {
  config(_input) {
    return {
      name: "nice-cks-graphrag",
      region: "eu-west-2",
    };
  },
  stacks(app) {
    app.stack(function API({ stack }) {
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
          "GET /health": "functions/health.handler",
        },
        defaults: {
          function: {
            runtime: "python3.11",
            layers: [layer],
            timeout: 30, // 30s for queries as per CLAUDE.md
            memorySize: 1024, // Start with 1024MB, adjust based on CloudWatch metrics
            environment: {
              MONGODB_URI: process.env.MONGODB_URI!,
              OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
              MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
              MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
              MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
              // Lambda-specific optimizations
              PYTHONPATH: "/opt/python:/var/task",
            },
          },
        },
        cors: {
          allowCredentials: true,
          allowHeaders: ["content-type", "authorization"],
          allowMethods: ["GET", "POST", "OPTIONS"],
          allowOrigins: ["*"], // Configure appropriately for production
        },
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
              MONGODB_URI: process.env.MONGODB_URI!,
              OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
              MONGODB_DB_NAME: process.env.MONGODB_DB_NAME || "ckshtn",
              MONGODB_GRAPH_COLLECTION: process.env.MONGODB_GRAPH_COLLECTION || "kg",
              MONGODB_VECTOR_COLLECTION: process.env.MONGODB_VECTOR_COLLECTION || "chunks",
              PYTHONPATH: "/opt/python:/var/task",
            },
          },
        },
      });
    });
  },
} satisfies SSTConfig;
