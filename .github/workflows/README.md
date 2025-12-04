# GitHub Actions Workflows

This directory contains automated workflows for the Movie Recommendation System.

## Available Workflows

### Azure Container Web App Deployment

**File:** `azure-container-webapp.yml`

**Purpose:** Automatically builds a Docker container and deploys it to Azure Container Web App.

**Triggers:**
- Push to `main` branch (automatic)
- Manual trigger via GitHub Actions UI

**Requirements Before First Run:**

1. **Azure Setup:**
   - Create Azure Container Web App (Linux, Docker)
   - Download Publish Profile from Azure Portal
   
2. **GitHub Secrets:**
   - Add `AZURE_WEBAPP_PUBLISH_PROFILE` secret
   
3. **GitHub PAT:**
   - Create token with `read:packages` and `write:packages` scopes
   
4. **Azure Web App Configuration:**
   - Add application settings for Docker registry
   - Set `WEBSITES_PORT` and `PORT` to `8000`
   
5. **Workflow Configuration:**
   - Update `AZURE_WEBAPP_NAME` in workflow file

**Documentation:**
- 📋 [Deployment Checklist](../../docs/DEPLOYMENT_CHECKLIST.md) - Quick setup guide
- 📖 [Full Documentation](../../docs/AZURE_DEPLOYMENT.md) - Detailed instructions
- 🐛 [Troubleshooting](../../docs/AZURE_DEPLOYMENT.md#troubleshooting) - Common issues

**How to Run Manually:**

1. Go to **Actions** tab
2. Select **Build and deploy a container to an Azure Web App**
3. Click **Run workflow** dropdown
4. Select branch (usually `main`)
5. Click **Run workflow** button

**Workflow Jobs:**

1. **build** - Builds Docker image and pushes to GitHub Container Registry
2. **deploy** - Deploys image to Azure Web App

**What Gets Deployed:**

- Docker image based on `Dockerfile`
- Container includes all application code, dependencies, and models
- Runs startup script `start.sh` which trains models if needed and starts the API

**After Deployment:**

Access your app at: `https://your-app-name.azurewebsites.net`
- API docs: `https://your-app-name.azurewebsites.net/docs`
- Health check: `https://your-app-name.azurewebsites.net/`

## Need Help?

- Check [Azure Deployment Guide](../../docs/AZURE_DEPLOYMENT.md)
- Review workflow logs in Actions tab
- Check Azure Web App logs in Azure Portal
- Open an issue in the repository
