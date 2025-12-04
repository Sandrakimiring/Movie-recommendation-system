# Azure Container Web App Deployment Guide

This guide explains how to deploy the Movie Recommendation System to Azure Container Web App using GitHub Actions.

## Prerequisites

Before deploying, you need:

1. **Azure Account**: Sign up at [portal.azure.com](https://portal.azure.com)
2. **Azure Container Web App**: Created in Azure Portal
3. **GitHub Repository**: Your fork or copy of this repository
4. **GitHub Personal Access Token (PAT)**: For GitHub Container Registry

## Step-by-Step Deployment Instructions

### 1. Create Azure Container Web App

1. Go to [Azure Portal](https://portal.azure.com)
2. Click **Create a resource** → **Web App**
3. Configure the web app:
   - **Name**: Choose a unique name (e.g., `movie-recs-app`)
   - **Publish**: Select **Docker Container**
   - **Operating System**: Select **Linux**
   - **Region**: Choose your preferred region
   - **Linux Plan**: Create new or select existing
4. In the **Docker** tab:
   - **Options**: Single Container
   - **Image Source**: GitHub Container Registry (ghcr.io)
   - **Image and tag**: Will be configured via workflow
5. Click **Review + Create** → **Create**

### 2. Download Azure Publish Profile

1. Navigate to your Azure Web App in the portal
2. Go to **Overview** page
3. Click **Get publish profile** (or **Download publish profile**)
4. Save the downloaded `.PublishSettings` file

### 3. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Create the following secret:
   - **Name**: `AZURE_WEBAPP_PUBLISH_PROFILE`
   - **Value**: Paste the entire contents of the publish profile file you downloaded

### 4. Create GitHub Personal Access Token (PAT)

1. Go to GitHub **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **Generate new token** → **Generate new token (classic)**
3. Configure the token:
   - **Note**: `Azure Container Registry`
   - **Expiration**: Choose appropriate duration
   - **Scopes**: Select `read:packages` and `write:packages`
4. Click **Generate token**
5. **Important**: Copy the token immediately (it won't be shown again)

### 5. Configure Azure Web App Settings

1. Go to your Azure Web App in the portal
2. Navigate to **Configuration** → **Application settings**
3. Add the following settings:

   | Name | Value |
   |------|-------|
   | `DOCKER_REGISTRY_SERVER_URL` | `https://ghcr.io` |
   | `DOCKER_REGISTRY_SERVER_USERNAME` | Your GitHub username (e.g., `Sandrakimiring`) |
   | `DOCKER_REGISTRY_SERVER_PASSWORD` | Your GitHub PAT token from step 4 |
   | `WEBSITES_PORT` | `8000` |
   | `PORT` | `8000` |

4. Click **Save**

### 6. Update Workflow Configuration

Edit `.github/workflows/azure-container-webapp.yml`:

```yaml
env:
  AZURE_WEBAPP_NAME: your-app-name  # Replace with your Azure Web App name
```

Replace `your-app-name` with the actual name of your Azure Web App (e.g., `movie-recs-app`).

### 7. Enable GitHub Container Registry

GitHub Container Registry (GHCR) is used by default. Ensure:

1. Your repository has **Packages** enabled
2. GitHub Actions has permissions to publish packages:
   - Go to **Settings** → **Actions** → **General**
   - Under **Workflow permissions**, select **Read and write permissions**

### 8. Trigger Deployment

The workflow is configured to run on:
- **Push to main branch**: Automatically deploys when you push to `main`
- **Manual trigger**: Run from Actions tab

To trigger manually:
1. Go to **Actions** tab in GitHub
2. Select **Build and deploy a container to an Azure Web App**
3. Click **Run workflow** → **Run workflow**

### 9. Monitor Deployment

1. Go to **Actions** tab to watch the workflow run
2. The workflow has two jobs:
   - **build**: Builds and pushes Docker image to GHCR
   - **deploy**: Deploys the image to Azure Web App

### 10. Verify Deployment

Once deployment completes:

1. Get your app URL: `https://your-app-name.azurewebsites.net`
2. Access the API documentation: `https://your-app-name.azurewebsites.net/docs`
3. Test the health endpoint: `https://your-app-name.azurewebsites.net/`

## Troubleshooting

### Deployment Fails

**Issue**: Build job fails
- **Solution**: Check the Dockerfile builds locally: `docker build -t test .`
- **Solution**: Ensure all dependencies are in `requirements.txt`

**Issue**: Deploy job fails with authentication error
- **Solution**: Verify Azure Web App settings have correct GHCR credentials
- **Solution**: Regenerate GitHub PAT and update Azure settings

**Issue**: Container starts but app is unreachable
- **Solution**: Verify `WEBSITES_PORT` is set to `8000` in Azure Web App settings
- **Solution**: Check Azure Web App logs: **Deployment Center** → **Logs**

### Application Errors

**Issue**: App crashes on startup
- **Solution**: Check container logs in Azure Portal: **Monitoring** → **Log stream**
- **Solution**: Ensure models are trained or training logic is in startup script

**Issue**: Missing data files
- **Solution**: Verify required data files are in the repository or downloaded during startup
- **Solution**: Check `start.sh` script for data initialization

### Permissions Issues

**Issue**: GitHub Actions cannot push to GHCR
- **Solution**: Enable workflow permissions: **Settings** → **Actions** → **General** → **Read and write permissions**

**Issue**: Azure cannot pull from GHCR
- **Solution**: Verify PAT has `read:packages` scope
- **Solution**: Confirm Azure Web App has correct registry credentials

## Environment Variables

The application supports these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port the application listens on |
| `WEBSITES_PORT` | - | Azure-specific port configuration |

## Production Considerations

### Security
- Use Azure Key Vault for sensitive configuration
- Rotate GitHub PAT regularly
- Use managed identities when possible

### Performance
- Consider Azure App Service Plan scaling options
- Monitor application metrics in Azure Monitor
- Use Azure CDN for static content if needed

### Cost Optimization
- Use appropriate App Service Plan tier
- Configure auto-scaling based on demand
- Monitor resource usage in Azure Cost Management

## Additional Resources

- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [GitHub Actions for Azure](https://github.com/Azure/actions)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Azure Web Apps Deploy Action](https://github.com/Azure/webapps-deploy)

## Support

For issues specific to:
- **Azure**: Check [Azure Support](https://azure.microsoft.com/en-us/support/)
- **GitHub Actions**: Check [GitHub Actions Documentation](https://docs.github.com/en/actions)
- **This Application**: Open an issue in the repository
