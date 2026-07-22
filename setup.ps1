$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".venv")) {
    py -3.11 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
    torch==2.13.0 torchvision==0.28.0 `
    --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

@'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'@ | .\.venv\Scripts\python.exe -
