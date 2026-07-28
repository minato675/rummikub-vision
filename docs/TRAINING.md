# 로컬 Teacher → Student 학습

## 의미

- Teacher: `rummikub-p8akb/2016` Roboflow 모델. 자동 라벨 초안을 생성합니다.
- Human review: Studio에서 클래스와 박스를 사람이 확인·수정합니다.
- Student: `yolo26n.pt`에서 시작하는 별도 Ultralytics YOLO26n 모델입니다.
- 두 번째 반복부터는 첫 번째 결과의 `best.pt`를 시작 가중치로 지정합니다.

Roboflow Docker 캐시는 추론 전용이므로 Teacher의 내부 가중치를 직접 수정하지 않습니다. Student는 검수된 Teacher 라벨을 학습해 해당 모델의 지식을 전달받습니다.

## 성능 기준

기본 합격 조건은 5-fold 교차검증 평균 `mAP50 >= 0.90`입니다. Precision, Recall, mAP50-95도 함께 저장됩니다. 90%는 데이터로 검증하는 목표이며 프로그램 설치만으로 보장되는 값은 아닙니다.

최종 성능 측정 시 Studio에서 fold를 `all`로 선택합니다. 빠른 개발 반복에서는 fold `1`만 사용합니다.

## 데이터 합치기

라벨이 없는 원본 이미지 폴더를 Teacher로 일괄 자동 라벨링할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe auto_label_folder.py C:\원본이미지 --prefix member1 --confidence 0.35
```

생성된 결과는 초안이므로 반드시 Studio에서 사람이 확인하고 저장한 뒤 학습해야 합니다.

각 조원은 이미지와 라벨을 같은 이름으로 준비합니다.

```text
받은데이터/
  images/all/example.jpg
  labels/all/example.txt
```

프로젝트 루트에서 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe import_dataset.py C:\받은데이터 --prefix member1
```

파일명이 충돌하면 prefix가 붙습니다. 데이터와 모델 파일은 Git에 올리지 않고 공유 드라이브나 Git LFS/DVC를 사용하세요.

## 90% 미달 시 순서

1. `rare_classes_under_20`에 나온 클래스부터 실제 카메라 환경에서 추가 수집합니다.
2. 낮은 신뢰도, 오검출, 반사·가림·회전 장면을 우선 검수합니다.
3. 이전 `best.pt`를 시작 가중치로 지정해 전체 데이터로 다시 학습합니다.
4. 같은 고정 seed와 5-fold 결과로 이전 버전과 비교합니다.

검증 fold의 이미지를 보고 라벨을 고친 뒤 같은 결과를 최종 성능으로 보고하면 데이터 누수가 됩니다. 수정 후에는 새 버전으로 전체 교차검증을 다시 실행하세요.
