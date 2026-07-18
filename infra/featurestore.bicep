// featurestore.bicep — resource-group-scoped (P9 A-3)
// Managed feature store workspace + its plain storage account.
// Borrows A0 Key Vault, ADLS Gen2 (offline store), UAMI; reuses P9 App Insights.

@description('Azure region.')
param location string = 'uksouth'

@description('Existing A0 Key Vault name.')
param keyVaultName string = 'kv-clariv-2v2zea7e6dd42'

@description('Existing A0 ADLS Gen2 account name.')
param adlsAccountName string = 'stclariv2v2zea7e6dd42'

@description('Existing A0 user-assigned managed identity name.')
param uamiName string = 'id-clariv-shared'

@description('Existing P9 Application Insights name.')
param appInsightsName string = 'appi-clariv-p9'

param tags object = {
  project: 'p9-cloud-native-ml'
  stage: 'stage2-featurestore'
  owner: 'estthorpe'
  costCenter: 'personal-portfolio'
}

var suffix = uniqueString(resourceGroup().id)
var fsStorageName = take('stfs${suffix}', 24)
var featureStoreName = 'fs-clariv-p9'

// ── EXISTING (look up, do not manage) ───────────────────────────────
resource keyVault 'Microsoft.KeyVault/vaults@2024-11-01' existing = {
  name: keyVaultName
}

resource adls 'Microsoft.Storage/storageAccounts@2024-01-01' existing = {
  name: adlsAccountName
}

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: uamiName
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: appInsightsName
}

// ── NEW: plain storage for the feature-store workspace ──────────────
resource fsStorage 'Microsoft.Storage/storageAccounts@2024-01-01' = {
  name: fsStorageName
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

resource featureStore 'Microsoft.MachineLearningServices/workspaces@2024-10-01' = {
  name: featureStoreName
  location: location
  tags: tags
  kind: 'FeatureStore'
  identity: {
    type: 'SystemAssigned, UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    friendlyName: 'Clariv P9 Feature Store'
    storageAccount: fsStorage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    featureStoreSettings: {
      offlineStoreConnectionName: 'offlineStoreConnectionName'
    }
  }
}

output featureStoreName string = featureStoreName
output fsStorageName string = fsStorage.name
