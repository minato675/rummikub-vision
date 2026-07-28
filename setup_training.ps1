$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv-train\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -3.11 -m venv (Join-Path $Root ".venv-train")
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip 업그레이드에 실패했습니다." }

# NVIDIA GPU용 PyTorch입니다. CPU 전용 PC는 아래 줄의 --index-url을 제거하세요.
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch 설치에 실패했습니다." }

& $Python -m pip install -r (Join-Path $Root "requirements-training.txt")
if ($LASTEXITCODE -ne 0) { throw "학습 패키지 설치에 실패했습니다." }

& $Python -c "import torch, ultralytics; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'ultralytics=', ultralytics.__version__)"
if ($LASTEXITCODE -ne 0) { throw "학습 환경 검증에 실패했습니다." }

Write-Host "학습 환경 설정 완료: $Python"
