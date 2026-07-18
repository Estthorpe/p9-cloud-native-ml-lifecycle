// compute.bicep — resource-group-scoped (P9 B-1)
// CPU training cluster: low-priority, min-0, quota-aware (2 vCPUs).

@description('Existing AML workspace name.')
param workspaceName string = 'mlw-clariv-p9'

@description('Azure region.')
param location string = 'uksouth'

// ── EXISTING (look up, do not manage) ───────────────────────────────
resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-10-01' existing = {
  name: workspaceName
}

// ── NEW: the training cluster (child of the workspace) ──────────────
resource cluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-10-01' = {
  parent: workspace
  name: 'cpu-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS2_v2'
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 1
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
    }
  }
}

output clusterName string = 'cpu-cluster'
