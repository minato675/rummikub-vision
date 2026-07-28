$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running. Start Docker Desktop and try again."
}

docker compose -f (Join-Path $ProjectRoot "compose.yaml") up -d
if ($LASTEXITCODE -ne 0) {
    throw "Could not start the Roboflow Inference container."
}

Write-Host "Roboflow Inference is starting at http://127.0.0.1:9001"
Write-Host "Run .\.venv\Scripts\python.exe wait_for_server.py to check readiness."

