# Rummikub Teacher–Student AI Studio

강한 Roboflow 공개 모델을 로컬 Teacher로 사용해 자동 라벨링하고, 사람이 검수한 데이터로 YOLO26n Student를 로컬 GPU에서 학습하는 팀 프로젝트입니다. 웹캠 탐지, 라벨 수정, 모델 테스트, 루미큐브 배치 규칙 판정, 80/20 교차검증 및 90% 목표 판정을 한 Studio에서 제공합니다.

## 빠른 설치 (Windows + NVIDIA)

```powershell
Copy-Item .env.example .env
# .env에 ROBOFLOW_API_KEY 입력
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\setup_training.ps1
powershell -ExecutionPolicy Bypass -File .\start_server.ps1
.\.venv\Scripts\python.exe studio.py
```

브라우저에서 `http://127.0.0.1:7861`을 엽니다. Docker Desktop과 NVIDIA 드라이버가 필요합니다.

권장 흐름은 `Teacher 자동 라벨 → 사람 검수 → 데이터 품질 검사 → fold 1 빠른 학습 → fold all 최종 평가`입니다. 기본 통과 기준은 교차검증 평균 `mAP50 >= 0.90`이며 설치만으로 보장되는 값이 아니라 데이터로 검증되는 목표입니다.

세부 학습 및 팀 데이터 병합 방법은 [docs/TRAINING.md](docs/TRAINING.md)를 확인하세요. `.env`, 이미지/라벨, 모델 가중치와 학습 결과는 Git에서 제외됩니다.

공개 모델 `rummikub-p8akb/2016`을 Roboflow의 GPU Inference Server로 이 PC에서 실행합니다. 최초 요청 때 모델이 Docker 볼륨에 내려받아지고 이후 프레임 추론은 `localhost:9001`에서 처리됩니다.

## 구성

```text
웹캠/OpenCV → localhost:9001 → GPU Inference Server → rummikub-p8akb/2016
```

기존 YOLO 직접 학습 프로젝트는 `../archive/yolo26_custom_training_20260723/`에 보존되어 있습니다.

## 1. Python 환경

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`.env.example`을 `.env`로 복사하고 `ROBOFLOW_API_KEY`를 입력합니다.

## 2. Docker Desktop과 로컬 서버

Docker Desktop을 실행한 다음:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_server.ps1
.\.venv\Scripts\python.exe wait_for_server.py
```

최초 모델 요청에서는 모델 다운로드와 초기화 때문에 시간이 더 걸릴 수 있습니다. 캐시는 Docker 볼륨 `rummikub-roboflow-model-cache`에 유지됩니다.

서버 중지:

```powershell
powershell -ExecutionPolicy Bypass -File .\stop_server.ps1
```

## 3. 이미지 테스트

```powershell
.\.venv\Scripts\python.exe infer_image.py C:\path\to\image.jpg
```

결과는 기본적으로 `outputs/inference.jpg`에 저장됩니다.

## 4. 웹캠

카메라 검색:

```powershell
.\.venv\Scripts\python.exe webcam.py --list-cameras
```

실행:

```powershell
.\.venv\Scripts\python.exe webcam.py --camera 0 --confidence 0.35
```

- `q` 또는 `Esc`: 종료
- `[` / `]`: 검색된 이전·다음 카메라로 전환

## 5. 로컬 Studio

```powershell
.\.venv\Scripts\python.exe studio.py --server-port 7861
```

Studio 기능:

- 로컬 Roboflow 모델 자동 라벨 초안
- 클래스 변경, 박스 삭제, 수동 두 모서리 박스
- 객체를 한 번 클릭해 모델이 찾은 경계 박스 적용
- 사람이 검수한 YOLO 라벨을 `data/images/all`, `data/labels/all`에 저장
- 단일 이미지 모델 테스트
- 저장 라벨 기준 Precision/Recall/F1 간이 평가
- 현재 테이블의 그룹·연속 배치 규칙 검사
- 회전된 보드에서도 인접 타일 방향을 자동 추정해 세트 군집화
- Roboflow 업로드용 ZIP 생성

## 6. 추가 학습

로컬에 캐시된 공개 모델은 추론용이므로 이 프로젝트가 그 가중치를 직접 수정하지 않습니다.

1. Universe의 `rummikub-p8akb` 프로젝트를 본인 Workspace로 Fork합니다.
2. Studio에서 만든 ZIP을 본인 프로젝트에 업로드합니다.
3. 클래스 매핑과 라벨을 확인하고 Dataset Version을 생성합니다.
4. Roboflow Train에서 새 모델 버전을 학습합니다. 제공되는 경우 기존 모델 checkpoint/transfer learning 옵션을 선택합니다.
5. 새 모델 ID를 `.env`의 `ROBOFLOW_MODEL_ID`에 입력합니다.
6. 로컬 Inference 서버와 Studio를 다시 시작합니다.

원본 공개 모델은 변경되지 않으며 개선 모델은 본인 Workspace에 생성됩니다.

## 주의

- API 키는 `.env`에만 저장하고 Git에 올리지 않습니다.
- 로컬 서버는 기본적으로 인증과 TLS가 없으므로 포트 9001을 외부 인터넷에 공개하지 않습니다.
- 최초 모델 다운로드에는 인터넷이 필요합니다.
- 모델 캐시는 추론용이며 YOLO 학습 체크포인트로 사용하는 파일이 아닙니다.
