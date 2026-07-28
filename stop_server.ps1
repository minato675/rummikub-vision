$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
docker compose -f (Join-Path $ProjectRoot "compose.yaml") stop

