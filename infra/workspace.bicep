// workspace.bicep — resource-group-scoped (P9)
// Azure ML workspace + its four support resources, deployed into the
// existing A0 resource group. References A0 Key Vault as existing.

@description('Azure region for all resources.')
param location string = 'uksouth'

@description('Short lowercase prefix used to build names.')
param namePrefix string = 'clariv'

@description('Name of the existing A0 Key Vault.')
param keyVaultName string = 'kv-clariv-2v2zea7e6dd42'

@description('Common tags applied to every resource.')
param tags object = {
  project: 'p9-cloud-native-ml'
  stage: 'stage1-template'
  owner: 'estthorpe'
  costCenter: 'personal-portfolio'
}

var suffix = uniqueString(resourceGroup().id)
var wsStorageName = take('stml${namePrefix}${suffix}', 24)
var logAnalyticsName = 'log-${namePrefix}-p9'
var appInsightsName = 'appi-${namePrefix}-p9'
var acrName = take('acr${namePrefix}${suffix}', 24)
var workspaceName = 'mlw-${namePrefix}-p9'

// ── EXISTING (A0 — look up, do not manage) ──────────────────────────
resource keyVault 'Microsoft.KeyVault/vaults@2024-11-01' existing = {
  name: keyVaultName
}

// ── NEW: plain storage for the workspace (filing cabinet, HNS OFF) ──
resource wsStorage 'Microsoft.Storage/storageAccounts@2024-01-01' = {
  name: wsStorageName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    isHnsEnabled: false
  }
}

// ── NEW: Log Analytics workspace (telemetry back-end) ───────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ── NEW: Application Insights (telemetry front-end) ─────────────────
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ── NEW: container registry (endpoint Docker images) ────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

// ── YOUR BLOCK: the Azure ML workspace ──────────────────────────────
// Type: 'Microsoft.MachineLearningServices/workspaces@2024-10-01'
// Needs: name (var workspaceName), location, tags,
//        identity (system-assigned),
//        properties wiring the four dependencies + a friendlyName.
// Property names Azure expects inside properties:
//   storageAccount, keyVault, applicationInsights, containerRegistry
//   (each takes a resource ID)


resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'Clariv P9 Workspace'
    storageAccount: wsStorage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id
  }
}

// ── Outputs ──────────────────────────────────────────────────────────
output workspaceName string = workspaceName
output wsStorageName string = wsStorage.name
output appInsightsName string = appInsights.name
output acrName string = acr.name
