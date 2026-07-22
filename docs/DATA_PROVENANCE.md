# 데이터 출처와 변환 기록

## Roboflow Universe 데이터

- 프로젝트: Rummikub
- 프로젝트 ID: `rummikub-p8akb`
- Workspace: `arthurvanmeerbeeck-gmail-com`
- Dataset version: `2017`
- URL: <https://universe.roboflow.com/arthurvanmeerbeeck-gmail-com/rummikub-p8akb>
- 라이선스: CC BY 4.0
- 로컬 가져오기 결과: 이미지 1,255장, 박스 13,065개

Roboflow가 반환한 YOLO 내보내기 라벨은 일부 객체에 다각형 좌표를 포함하고
있어, 각 다각형의 `min(x), min(y), max(x), max(y)`로 외접 Bounding Box를
계산했습니다. 원본 클래스 순서는 내보낸 `data.yaml`에서 읽은 뒤 로컬 53개
클래스 순서로 다시 매핑했습니다.

정확한 가져오기 시각, split별 개수, 전체 클래스 매핑은
`data/roboflow_import_manifest.json`에 기록됩니다.

## 학습 시 주의

공개 데이터는 초기 모델 구축을 위한 bootstrap 데이터입니다. 실제 게임
카메라의 조명, 거리, 테이블 배경과 분포가 다를 수 있으므로 사용 환경에서
직접 수집하고 사람이 검수한 이미지를 추가해야 합니다. 자동 라벨은 ground
truth가 아니며 학습 전에 반드시 검수해야 합니다.
