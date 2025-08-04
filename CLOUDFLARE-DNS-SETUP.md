# Cloudflare DNS Setup - Quick Reference

## 🚀 Quick Setup for graphrag.care

### Step 1: Add DNS Records in Cloudflare

Log into Cloudflare → Select graphrag.care → DNS → Add these records:

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| CNAME | api | api.graphrag.care | ✅ ON |
| CNAME | staging-api | staging-api.graphrag.care | ✅ ON |
| A | @ | 192.0.2.1 | ✅ ON |

### Step 2: Configure Page Rule

Create page rule: `graphrag.care/*` → 301 Redirect to `https://api.graphrag.care/$1`

### Step 3: SSL Settings

Go to SSL/TLS → Set to "Full (strict)"

### Step 4: Test Your New URLs

```bash
# Test production
curl https://api.graphrag.care/health

# Test staging  
curl https://staging-api.graphrag.care/health
```

## 📋 Current API Endpoints

| Environment | Old URL | New URL |
|-------------|---------|---------|
| Production | https://api.graphrag.care | https://api.graphrag.care |
| Staging | https://staging-api.graphrag.care | https://staging-api.graphrag.care |

## 🔄 DNS Propagation

After adding records, DNS propagation typically takes:
- With Cloudflare Proxy: 1-5 minutes
- Global propagation: Up to 24 hours

Check propagation: https://www.whatsmydns.net/#CNAME/api.graphrag.care

## ✅ Verification Checklist

- [ ] CNAME records added for api and staging-api
- [ ] Cloudflare proxy (orange cloud) enabled
- [ ] SSL set to "Full (strict)"
- [ ] Page rule created for root domain redirect
- [ ] Test endpoints return 200 OK
- [ ] Update frontend to use new URLs

## 🆘 Troubleshooting

If domains don't work after 10 minutes:
1. Verify CNAME targets are exactly as shown above
2. Ensure proxy (orange cloud) is ON
3. Check SSL/TLS settings are "Full (strict)"
4. Clear DNS cache: `sudo dscacheutil -flushcache` (macOS)

---
*Created: August 3, 2025*