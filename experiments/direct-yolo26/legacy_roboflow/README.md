# Rummikub Tile Detector

Roboflow Universe의 `rummikub-p8akb/2016` 모델을 호출해 이미지 속 루미큐브
타일을 검출하는 Python CLI입니다. 53개 클래스(검정/파랑/주황/빨강 1~13 및
Joker)를 지원합니다.

## 설치 (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` 파일의 `ROBOFLOW_API_KEY`를 본인의 Roboflow private API key로
교체합니다. API key는 Roboflow의 **Settings > API Keys**에서 확인할 수 있습니다.

## 실행

```powershell
.venv\Scripts\python.exe app.py path\to\rummikub.jpg
```

기본 결과는 `outputs/result.jpg`와 `outputs/result.json`에 저장됩니다.

```powershell
.venv\Scripts\python.exe app.py sample.jpg `
  --confidence 0.5 `
  --output outputs/annotated.jpg `
  --json-output outputs/predictions.json
```

도움말은 다음과 같이 확인합니다.

```powershell
.venv\Scripts\python.exe app.py --help
```

## 웹캠 실시간 탐지

```powershell
.venv\Scripts\python.exe webcam.py
```

창에서 `q` 또는 `Esc`를 누르면 종료됩니다. `c`를 누르면 사용 가능한 다음
카메라로 전환되고, 숫자 키 `0`~`9`로 원하는 카메라 번호를 바로 선택할 수
있습니다. `+`와 `-` 키로 표시할 최소 신뢰도를 5%씩 조절합니다.

연결된 카메라 번호 확인:

```powershell
.venv\Scripts\python.exe webcam.py --list-cameras
```

시작할 카메라나 호출 간격도 옵션으로 지정할 수 있습니다.

```powershell
.venv\Scripts\python.exe webcam.py --camera 1 --interval 0.7 --confidence 0.5
```

`--interval`은 Roboflow에 추론 프레임을 보내는 최소 간격(초)입니다. 값을
줄이면 탐지 반응은 빨라지지만 API 사용량이 증가합니다. 전체 옵션은
`.venv\Scripts\python.exe webcam.py --help`로 확인할 수 있습니다.

재학습한 모델을 사용하려면 `.env`의 모델 ID를 새 배포 ID로 변경합니다.

```dotenv
ROBOFLOW_MODEL_ID=your-project-id/version-number
```

## 참고

- 추론은 Roboflow Serverless API에서 실행되므로 인터넷 연결과 API key가 필요합니다.
- 웹캠 영상에서 선택된 프레임이 추론을 위해 Roboflow 서버로 전송됩니다.
- API key가 없거나 호출 한도를 초과하면 추론할 수 없습니다.
- 원본 데이터셋은 CC BY 4.0 라이선스로 공개되어 있습니다.
