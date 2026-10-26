# Rummikub Vision

저는 루미큐브 인식·라벨링·학습·규칙 검증 실험을 두 가지 방식으로 나누어 정리했습니다.
각 실험의 기존 코드와 커밋 이력을 보존하고, 실행 환경과 검증 범위는 실험별 README에 남겼습니다.

| 실험 | 내용 |
| --- | --- |
| [직접 YOLO26 학습](experiments/direct-yolo26/README.md) | 공개 라벨 가져오기·검수, MobileSAM 스마트 박스, YOLO26n 학습과 턴 규칙 검증 |
| [Teacher–Student](experiments/teacher-student/README.md) | 로컬 Roboflow Teacher 자동 라벨, 사람 검수, Student 학습·교차검증·웹캠 추론 |

각 실험 폴더로 이동한 뒤 해당 README의 설치·실행 절차를 따릅니다.
`.env`, 이미지·라벨 데이터, 모델 가중치와 학습 결과는 별도로 준비해야 합니다.
Teacher–Student 문서의 `mAP50 >= 0.90`은 목표 기준이며, 달성한 실측 성과를 의미하지 않습니다.

이 저장소는 서로 다른 실험을 함께 관리하기 위한 통합입니다. 두 구현을 하나의 검증된 모델이나
동일한 학습 파이프라인으로 합쳤다고 설명하지 않습니다.

[원본 이력과 통합 기록](docs/REPOSITORY_MERGE.md)
