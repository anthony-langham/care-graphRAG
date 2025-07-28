# Lambda Layers

This directory contains the Python dependencies layer for AWS Lambda functions.

## Structure

- `python/requirements.txt` - Lambda layer dependencies (optimized subset of main requirements)
- `python/build_layer.sh` - Build script to create the Lambda layer package

## Building the Layer

To build the Lambda layer locally:

```bash
cd layers/python
./build_layer.sh
```

This will:
1. Install all dependencies from `requirements.txt` into a `python/` directory
2. Clean up unnecessary files to reduce layer size
3. Create `python-deps.zip` for deployment

## Deployment

The layer is automatically deployed by SST when running:

```bash
sst deploy
```

SST references the layer in `sst.config.ts`:

```typescript
const layer = new LayerVersion(stack, "PythonDeps", {
  code: Code.fromAsset("layers/python"),
  compatibleRuntimes: [Runtime.PYTHON_3_11],
});
```

## Layer Optimization

The requirements.txt is optimized for Lambda:
- Includes only runtime dependencies
- Excludes heavy visualization libraries (networkx, plotly, pandas)
- Excludes development/testing dependencies
- Total size should be under 50MB compressed

## Updating Dependencies

When updating dependencies:
1. Update `layers/python/requirements.txt`
2. Run `./build_layer.sh` locally to test
3. Deploy with `sst deploy`

Note: Lambda has a 250MB uncompressed layer size limit.