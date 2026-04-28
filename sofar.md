# LODO 실험 진행 현황 정리 (so far)

작성일: 2026-04-27  
정리 기준: `RGB-master` 폴더 내 코드/결과 파일 기준

---

## 0) 프로젝트 목적(공통)

- RAG 벤치마크 환경에서 LLM의 정답성/거절 성향/반사실 강건성/문서 인과적 중요도를 평가.
- 특히 문서 제거(LODO: Leave-One-Document-Out) 시 모델의 출력 확률과 내부 표현(히든 스테이트)이 어떻게 변하는지 기계론적으로 분석.

참고 파일:
- `RGB-master/readme.md`
- `RGB-master/evalue.py`
- `RGB-master/run_lodo_experiments.py`

---

## 1) 기본 성능 체크 실험 (single run)

### 실험 목적
- 커스텀 스타일 데이터셋(`en_mid_conversational`)에서 Llama3의 기본 정확도 확인.

### 실험 방법
- 실행/결과 파일명 기준 설정:
  - dataset: `en_mid_conversational`
  - model: `llama3`
  - temperature: `0.6`
  - noise_rate: `0.0`
  - passage_num: `5`
  - correct_rate: `0.0`
- 평가 스크립트 구조: `evalue.py` (`all_rate = 정답 판정 비율`)

### 실험 결과
- 결과 파일: `RGB-master/result-en/prediction_en_mid_conversational_llama3_temp0.6_noise0.0_passage5_correct0.0_result.json`
- `all_rate = 0.92` (46/50)

---

## 2) Counter-factual 다중 실행 실험 (스타일별, 10회 반복)

### 실험 목적
- 스타일 프롬프트(Base/Academic/Confident/Conversational/Narrative)가 반사실 상황에서 성능(정확/거절/오답)에 주는 영향을 비교.
- 실행 랜덤성을 줄이기 위해 스타일별 반복 실행(10회) 후 평균/표준편차로 비교.

### 실험 방법
- 데이터셋: `en_counter_mid*` 계열 (스타일별)
- 모델/파라미터(파일명 기준): `llama3`, `temp=0.6`, `passage=5`, `correct=0.0`
- 노이즈 조건 2개:
  - `noise-0` (0.0)
  - `noise-0.4` (0.4)
- 각 노이즈 조건에서 스타일별 `run1~run10` 결과 저장.
- 총 실행 수:
  - 노이즈당 50개 결과 파일(스타일 5개 x 10회)
  - 각 실행의 평가 샘플 수 `nums=50`

요약 파일:
- `RGB-master/result-en/multiple-runs/counter-factual/noise-0/summary.json`
- `RGB-master/result-en/multiple-runs/counter-factual/noise-0.4/summary.json`

### 실험 결과 (요약)

#### noise = 0.0
- Accuracy 최고: `Narrative` (15.4)
- Reject 최고: `Conversational` (14.2)
- Wrong(오답률) 최저: `Conversational` (75.0)

#### noise = 0.4
- Accuracy 최고: `Conversational` (17.0)
- Reject 최고: `Academic` (13.0)
- Wrong(오답률) 최저: `Conversational` (70.6)

#### noise 0.0 -> 0.4 변화(평균)
- Base: Accuracy +4.0, Reject -4.6, Wrong +0.6
- Academic: Accuracy +4.0, Reject +0.2, Wrong -4.2
- Confident: Accuracy +4.2, Reject -3.8, Wrong -0.4
- Conversational: Accuracy +6.2, Reject -1.8, Wrong -4.4
- Narrative: Accuracy +0.6, Reject +1.4, Wrong -2.0

---

## 3) LODO 기계론 실험 (문서 제거 인과 분석)

### 실험 목적
- 질의별 문서를 1개씩 제거했을 때:
  - 정답성(fact score) 변화
  - baseline 답변의 로그확률 저하(logprob degradation)
  - 레이어별 표현 드리프트(L2)
를 측정해 인과적으로 중요한 문서를 찾기.

### 실험 방법
- 스크립트: `RGB-master/run_lodo_experiments.py`
- 설정:
  - dataset: `en_refine`
  - model: `llama3`
  - temp: 0.7(기본값)
- 절차:
  1. 전체 문서로 baseline 생성/채점.
  2. 문서 1개 제거 후 재생성 + baseline 답변 로그확률 재평가.
  3. `fact_degradation`, `logprob_degradation`, `representation_drift_l2` 기록.
  4. 기준: `fact_degradation > 0` 또는 `logprob_degradation < -2.0`이면 인과 중요.
- 실험 스크립트상 샘플 수는 데모 형태로 `처음 10개 질의`만 실행(`instances[:10]`).

결과 파일:
- `RGB-master/lodo_results_en_refine_llama3.json`

### 실험 결과 (집계)
- 질의 수: 10
- 총 문서 제거(ablation) 수: 275
- 평균 logprob degradation: `-0.0557`
- 최소/최대 logprob degradation: `-3.375` / `0.8125`
- fact_degradation 분포: `{0: 274, 1: 1}`
- 인과 중요 판정: `4 / 275 (1.45%)`

주요 케이스:
- 강한 확률 붕괴(사실 정답 유지):  
  - query id 2, doc 2, logprob -3.375  
  - query id 3, doc 0, logprob -2.2539  
  - query id 6, doc 0, logprob -2.125
- 사실성 직접 악화 케이스:  
  - query id 9, doc 3, fact_degradation=1

---

## 4) Detailed Case Study (Top divergence vs 평균 케이스)

### 실험 목적
- LODO에서 발견된 극단 케이스와 평균 케이스를 깊게 비교해,
  - 토큰별 로그확률 변화
  - 전체 레이어 드리프트 궤적
차이를 정밀 분석.

### 실험 방법
- 스크립트: `RGB-master/run_detailed_case_study.py`
- LODO 결과에서
  - Top divergence 5개(큰 음수 logprob, fact_deg=0 중심)
  - Average baseline 5개(logprob 변화가 0에 가까운 케이스)
를 선택.
- 각 케이스에 대해:
  - baseline vs ablated 토큰별 logprob
  - layer 0~32 드리프트
  추출.
- 결과 저장: `RGB-master/detailed_case_studies.json`

### 실험 결과 (집계)
- 총 케이스 수: 10
- Top Divergence (5개):
  - 평균 원 logprob degradation: `-2.132`
  - 마지막 레이어 평균 drift: `8.8535` (최대 `15.1069`)
- Average Baseline (5개):
  - 평균 원 logprob degradation: `0.0`
  - 마지막 레이어 평균 drift: `3.4330` (최대 `5.0288`)

해석:
- 확률 붕괴가 큰 케이스일수록 심층 레이어에서 표현 드리프트가 훨씬 크게 나타나는 경향이 확인됨.

---

## 5) 시각화 산출물

### 실험 목적
- LODO/Case-study 결과를 분포/상관/궤적으로 시각화해 패턴 확인.

### 실험 방법
- `visualize_lodo.py`, `visualize_case_study.py` 실행 시 `plots/`에 그림 저장.
- 박스플롯, 카운트플롯, 산점도, 토큰 궤적, 레이어 궤적 등 생성.

### 실험 결과
- 생성 대상 파일(스크립트 기준):
  - `plots/1_logprob_outliers.png`
  - `plots/2_layer_drift_outliers.png`
  - `plots/3_fact_degradation_counts.png`
  - `plots/4_logprob_by_importance.png`
  - `plots/5_mechanistic_divergence.png`
  - `plots/6_tokenwise_trajectory.png`
  - `plots/7_layerwise_drift_curve.png`

---

## 메모 / 확인 필요 사항

- `RGB-master/result-en/multiple-runs/counter-factual/statistical_analysis.json` 파일은 현재 내용이 중간에서 끊겨 있어(불완전 JSON) 통계 유의성 결과를 신뢰하기 어려움.
- 현재 요약은 저장된 파일 기준이므로, 미저장 실험/별도 노트가 있으면 추가 반영 가능.
