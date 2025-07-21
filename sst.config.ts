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
      });

      // API Lambda
      const api = new Api(stack, "api", {
        routes: {
          "POST /query": "functions/query.handler",
          "GET /health": "functions/health.handler",
        },
        defaults: {
          function: {
            runtime: "python3.11",
            layers: [layer],
            timeout: 30,
            memorySize: 1024,
            environment: {
              MONGODB_URI: process.env.MONGODB_URI,
              OPENAI_API_KEY: process.env.OPENAI_API_KEY,
            },
          },
        },
      });

      // Scheduled sync
      new Cron(stack, "sync", {
        schedule: "rate(7 days)",
        job: {
          function: {
            handler: "functions/sync.handler",
            layers: [layer],
            timeout: 300, // 5 minutes for sync
          },
        },
      });
    });
  },
} satisfies SSTConfig;