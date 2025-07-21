# MongoDB Atlas Security Configuration Guide

## TASK-006: Configure Atlas Security

This guide walks through the manual steps needed to configure MongoDB Atlas security settings.

## Steps to Complete

### 1. Create Database User

1. Log into MongoDB Atlas (https://cloud.mongodb.com)
2. Navigate to your project
3. Click "Database Access" in the left sidebar
4. Click "Add New Database User"
5. Configure the user:
   - Authentication Method: Password
   - Username: `ckshtn-admin`
   - Password: Generate a secure password (save this for .env)
   - Database User Privileges: "Atlas Admin" or "Read and write to any database"
   - Click "Add User"

### 2. Add Current IP to Whitelist

1. Click "Network Access" in the left sidebar
2. Click "Add IP Address"
3. Options:
   - For development: Click "Add Current IP Address"
   - For production: Add specific IP addresses of AWS Lambda NAT Gateway
4. Add a comment describing the IP (e.g., "Development machine" or "AWS Lambda eu-west-2")
5. Click "Confirm"

### 3. Enable IP Access List for Production

For Lambda deployment, you'll need to:

1. Determine your Lambda's egress IP addresses (NAT Gateway IPs)
2. Add those specific IPs to the access list
3. Consider using VPC Peering or AWS PrivateLink for more secure connectivity

### 4. Copy Connection String

1. Go to "Database" in the left sidebar
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Select:
   - Driver: Python
   - Version: 3.11 or later
5. Copy the connection string, it should look like:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

### 5. Update .env File

Create a `.env` file (copy from `.env.template`):

```bash
cp .env.template .env
```

Update with your values:

```env
OPENAI_API_KEY=sk-***
MONGODB_URI=mongodb+srv://ckshtn-admin:WK2JtVpHPsIriwnD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=ckshtn
MONGODB_GRAPH_COLLECTION=kg
MONGODB_VECTOR_COLLECTION=chunks
```

## Security Best Practices

1. **Use Strong Passwords**: Generate passwords with 16+ characters including special characters
2. **Principle of Least Privilege**: Create separate users for different environments
3. **IP Whitelisting**: Only allow known IPs, avoid 0.0.0.0/0 in production
4. **Connection String Security**: Never commit connection strings to git
5. **Regular Rotation**: Rotate database passwords periodically

## Next Steps

After completing these manual steps:

1. Run `scripts/test_connection.py` to verify connectivity (TASK-008)
2. Create the database and collections (TASK-007)
