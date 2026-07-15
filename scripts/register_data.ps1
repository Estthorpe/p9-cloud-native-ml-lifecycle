# register_data.ps1 — P9 A-2: validate, upload to ADLS, register versioned data assets.
# Fail-closed: contracts must pass before any upload or registration happens.

$ErrorActionPreference = "Stop"

# Station 1: the contract gate
python -m src.ingestion.contracts
if ($LASTEXITCODE -ne 0) {
    Write-Error "Data contracts FAILED - aborting. Nothing was uploaded."
    exit 1
}

# Station 2: upload both CSVs to the data lake (raw/ folder)
$dataDir = "C:/Users/enuzo/Documents/ai_eng_projs/predictive-maintenance-triage/data"
az storage fs file upload --file-system datalake --source "$dataDir/sensor_data_real.csv" --path raw/sensor_data_real.csv --account-name stclariv2v2zea7e6dd42 --auth-mode login --overwrite
az storage fs file upload --file-system datalake --source "$dataDir/anomaly_labels_real.csv" --path raw/anomaly_labels_real.csv --account-name stclariv2v2zea7e6dd42 --auth-mode login --overwrite

# Station 3: register versioned data assets pointing at the datastore paths
az ml data create --name sensor-data-real --version 1 --type uri_file --path "azureml://datastores/adls_datalake/paths/raw/sensor_data_real.csv" --resource-group rg-clariv-foundation --workspace-name mlw-clariv-p9 --description "NASA IMS bearing RMS vibration (8,624 rows, 1st_test, 4 bearings). Validated by src/ingestion/contracts.py."
az ml data create --name anomaly-labels-real --version 1 --type uri_file --path "azureml://datastores/adls_datalake/paths/raw/anomaly_labels_real.csv" --resource-group rg-clariv-foundation --workspace-name mlw-clariv-p9 --description "3-sigma control chart anomaly labels (65 anomalies, 0.8%). Validated by src/ingestion/contracts.py."

Write-Host "`nDone: 2 files uploaded, 2 data assets registered at version 1." -ForegroundColor Green
