# Azure Deployment Checklist

Use this checklist to ensure you've completed all steps for deploying to Azure Container Web App.

## Pre-Deployment Setup

- [ ] **Azure Account Created**
  - Sign up at [portal.azure.com](https://portal.azure.com)

- [ ] **Azure Container Web App Created**
  - Name: `_________________`
  - Region: `_________________`
  - Docker Container configured
  - Linux OS selected

- [ ] **Publish Profile Downloaded**
  - Downloaded from Azure Web App Overview page
  - File saved locally

## GitHub Configuration

- [ ] **Repository Secret Created**
  - Secret name: `AZURE_WEBAPP_PUBLISH_PROFILE`
  - Value: Contents of publish profile file pasted
  - Location: Repository Settings → Secrets and variables → Actions

- [ ] **GitHub PAT Created**
  - Token created at: Settings → Developer settings → Personal access tokens
  - Scopes: `read:packages` and `write:packages` selected
  - Token copied and saved securely

- [ ] **Workflow Permissions Configured**
  - Location: Repository Settings → Actions → General
  - Selected: "Read and write permissions"

## Azure Web App Configuration

- [ ] **Container Registry Settings Added**
  - Navigate to: Web App → Configuration → Application settings
  - Add these settings:
    - [ ] `DOCKER_REGISTRY_SERVER_URL` = `https://ghcr.io`
    - [ ] `DOCKER_REGISTRY_SERVER_USERNAME` = Your GitHub username
    - [ ] `DOCKER_REGISTRY_SERVER_PASSWORD` = Your GitHub PAT
    - [ ] `WEBSITES_PORT` = `8000`
    - [ ] `PORT` = `8000`
  - [ ] Clicked "Save" to apply settings

## Workflow Configuration

- [ ] **Workflow File Updated**
  - File: `.github/workflows/azure-container-webapp.yml`
  - Updated `AZURE_WEBAPP_NAME` with your actual Azure Web App name
  - Committed and pushed changes

## Deployment

- [ ] **Trigger Deployment**
  - Option A: Push to main branch (automatic)
  - Option B: Manual trigger from Actions tab

- [ ] **Monitor Deployment**
  - Watch GitHub Actions workflow run
  - Both jobs complete successfully:
    - [ ] Build job (pushes Docker image)
    - [ ] Deploy job (deploys to Azure)

## Verification

- [ ] **Test Deployment**
  - [ ] Access app URL: `https://your-app-name.azurewebsites.net`
  - [ ] API docs work: `https://your-app-name.azurewebsites.net/docs`
  - [ ] Health endpoint responds: `https://your-app-name.azurewebsites.net/`
  - [ ] Test a recommendation endpoint

- [ ] **Check Logs (if issues)**
  - [ ] Azure Portal: Web App → Monitoring → Log stream
  - [ ] GitHub Actions: Check workflow logs

## Post-Deployment

- [ ] **Document Your Setup**
  - Azure Web App Name: `_________________`
  - App URL: `_________________`
  - Deployment Date: `_________________`

- [ ] **Share Access**
  - Share app URL with team/users
  - Document any API keys or authentication

## Troubleshooting Resources

If you encounter issues, refer to:
- [Full Deployment Guide](AZURE_DEPLOYMENT.md)
- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [GitHub Actions Logs](../../actions)

---

**Need Help?**
- Review the detailed guide: [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md)
- Check Azure Web App logs in Azure Portal
- Review GitHub Actions workflow logs
- Open an issue in the repository
