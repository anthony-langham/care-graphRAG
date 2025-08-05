# Certificate DNS Validation Records for Cloudflare

Add these CNAME records to Cloudflare to validate your SSL certificates:

## 1. Production Certificate (api.graphrag.care)

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| CNAME | `_fd85366f8263617cbab1e3a62e8db3a4.api` | `_3e88546a0d498441dcf01bf813edfef7.xlfgrmvvlj.acm-validations.aws.` | OFF (gray cloud) |

## 2. Staging Certificate (staging-api.graphrag.care)

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| CNAME | `_49b9e4d26788db83ad30bb7498bec734.staging-api` | `_96ed26d5810c29eb77ff7f59e0f13874.xlfgrmvvlj.acm-validations.aws.` | OFF (gray cloud) |

## Important Notes:

1. **Proxy must be OFF** (gray cloud) for validation records
2. **Include the trailing dot** in the Content/Value field
3. Validation typically takes 5-30 minutes after adding records
4. These records must remain in place as long as you use the certificates

## Certificate ARNs (for SST config):

- **Production**: `arn:aws:acm:us-east-1:146409062658:certificate/83bb5fcb-dad5-4ad4-93a1-c935fd52a9f9`
- **Staging**: `arn:aws:acm:us-east-1:146409062658:certificate/841b8597-6088-4ca8-be38-12f72c185af6`

---
Generated: 2025-08-04