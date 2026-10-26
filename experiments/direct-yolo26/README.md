# Rummikub YOLO26 Studio

Ultralytics YOLO26 Nano(`yolo26n.pt`)를 기반으로 직접 모델을 학습하는 로컬
프로젝트입니다. CC BY 4.0 공개 라벨은 초기 데이터로만 사용하고, 기존
Roboflow 모델은 자동 라벨 초안을 만드는 선택 기능으로 분리했습니다.

주요 기능:

- 업로드 또는 웹캠 촬영 이미지 라벨링
- Roboflow Universe 공개 라벨 데이터 가져오기 및 검수
- Roboflow/로컬 YOLO26 자동 라벨 초안 생성
- MobileSAM 기반 한 번 클릭 스마트 Bounding Box
- 53개 루미큐브 클래스의 YOLO Detect 형식 저장
- Train/Validation/Test 데이터 분리와 데이터 검증
- RTX GPU를 이용한 YOLO26n 전이학습
- 이전 턴과 현재 턴 이미지의 타일 검출 결과 비교
- 그룹, 연속, 최초 등록 30점, 테이블 타일 유실 규칙 검증
- 위반 시 마지막 정상 턴 이미지 반환

기존 Roboflow 구현은 `legacy_roboflow/`에 보존되어 있으며 새 시스템에서는
사용하지 않습니다.

## 1. 환경 구성

현재 PC에는 Python 3.11, PyTorch CUDA 13.0, RTX 4050 환경이 이미 구성되어
있습니다. 다시 구성할 때는 PowerShell에서 실행합니다.

```powershell
.\setup.ps1
```

## 2. 라벨링 및 턴 검증 UI

```powershell
.\.venv\Scripts\python.exe studio.py
```

브라우저에서 세 개 탭을 사용할 수 있습니다.

1. **데이터 라벨링**: 기존 라벨 검수, 자동 초안, 한 번 클릭 스마트 박스 및 수동 박스
2. **턴 룰 검증**: 기준 턴 저장 → 현재 턴 촬영 → 판정 및 정상 턴 확정
3. **학습 안내**: 데이터 검사 및 학습 명령 확인

라벨은 `data/images/{train,val,test}`와 `data/labels/{train,val,test}`에
YOLO 형식으로 저장됩니다. 타일이 전혀 없는 배경 이미지도 별도 체크 후
negative sample로 저장할 수 있습니다.

라벨링 도구:

- **스마트 박스**: 클래스를 고르고 타일을 한 번 클릭하면 MobileSAM이 경계를 생성
- **수동 박스**: 타일의 반대쪽 두 모서리를 차례로 클릭
- **클래스 변경**: 새 클래스를 고른 뒤 기존 박스 내부 클릭
- **박스 삭제**: 삭제 모드에서 기존 박스 내부 클릭
- **자동 라벨링**: Roboflow 기존 모델 또는 학습된 로컬 YOLO26n으로 초안 생성
- **기존 라벨 검수**: 저장된 이미지를 불러와 수정 후 원본 라벨에 덮어쓰기

자동 라벨 결과는 정답이 아니므로 반드시 사람이 확인한 뒤 저장해야 합니다.

### 기존 공개 라벨 데이터

Roboflow Universe의 `rummikub-p8akb/2017` 버전을 가져와 YOLO Detect 형식으로
변환했습니다.

| 분할 | 이미지 | 박스 |
|---|---:|---:|
| Train | 1,095 | 11,249 |
| Validation | 107 | 1,179 |
| Test | 53 | 637 |
| 합계 | 1,255 | 13,065 |

원본 다각형 좌표는 외접 Bounding Box로 변환했고 `Black10` 같은 클래스명은
`black_10` 형식으로 통일했습니다. 다시 가져오려면 실행합니다.

```powershell
.\.venv\Scripts\python.exe import_roboflow.py
```

출처와 변환 기록은 [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) 및
`data/roboflow_import_manifest.json`에서 확인할 수 있습니다.

자동 라벨링에서 Roboflow를 선택하면 이미지가 API 서버로 전송됩니다. 로컬
YOLO26n과 MobileSAM은 이 PC의 GPU에서 실행됩니다.

## 3. 데이터 수집 기준

- 클래스: `black_1`~`black_13`, `blue_1`~`blue_13`,
  `orange_1`~`orange_13`, `red_1`~`red_13`, `joker`
- 이미지에 보이는 모든 타일을 빠짐없이 라벨링합니다.
- 박스는 타일 외곽에 밀착시킵니다.
- 조명, 배경, 거리, 각도, 그림자, 반사가 다른 장면을 포함합니다.
- 연속 프레임을 무작정 저장하지 말고 서로 다른 장면을 수집합니다.
- 같은 촬영 세션의 유사 이미지는 한 split에만 넣어 데이터 누수를 막습니다.
- 처음에는 클래스마다 최소 50개 이상을 목표로 하되, 검증 결과가 낮은
  클래스를 우선 보강합니다.

## 4. 데이터 검사

```powershell
.\.venv\Scripts\python.exe validate_dataset.py
```

이미지-라벨 쌍, 클래스 번호, 정규화 좌표 범위를 검사합니다.

## 5. YOLO26 Nano 학습

```powershell
.\.venv\Scripts\python.exe train.py --epochs 100 --imgsz 960
```

기본 모델은 COCO 사전학습 `yolo26n.pt`이며 Roboflow의 학습된 가중치를
가져오지는 않습니다. 공개 라벨과 직접 검수한 라벨로 새 모델을 학습합니다.
결과 가중치는 다음에 저장됩니다.

```text
runs/detect/rummikub_yolo26n/weights/best.pt
```

중단된 학습 재개:

```powershell
.\.venv\Scripts\python.exe train.py `
  --resume runs\detect\rummikub_yolo26n\weights\last.pt
```

## 6. 턴 판정 사용 조건

- 카메라는 테이블 위에 고정하고 턴 사이에 이동하지 않습니다.
- 기준 턴과 현재 턴에서 전체 테이블이 모두 보여야 합니다.
- 타일 세트 사이 간격을 타일 내부 간격보다 충분히 크게 둡니다.
- 플레이어별 최초 등록 완료 여부는 UI 체크박스로 관리합니다.
- 모델 오검출 가능성이 있으므로 대회 심판을 완전히 대체하는 용도가 아니라
  판정 보조 시스템으로 사용해야 합니다.

세부 규칙과 현재 자동 판정의 한계는 [docs/RULES_AND_LIMITATIONS.md](docs/RULES_AND_LIMITATIONS.md)를
참고하세요.

## 라이선스 주의

Ultralytics 저장소는 AGPL-3.0 및 Enterprise 라이선스 옵션을 제공합니다.
폐쇄형 또는 상용 서비스로 배포할 예정이라면 사용 방식에 맞는 라이선스를
반드시 확인해야 합니다.
