# Wafer WTMS (Wavelet Transform Modulus Sum) Defect Detection

**웨이퍼 표면 결함 검출**을 위한 WTMS(Wavelet Transform Modulus Sum) 기반 파이프라인입니다.
고해상도 웨이퍼 이미지에서 **스크래치**와 **파티클** 결함을 정확하게 검출하고 분류합니다.

본 구현체는 다음 IEEE 논문에 기반합니다:
> *"A Wavelet-Based Approach in Detecting Visual Defects on Semiconductor Wafer Dies,"* in IEEE Transactions on Semiconductor Manufacturing, vol. 23, no. 2, pp. 284-292, May 2010, doi: 10.1109/TSM.2010.2046108

---

## 1. 프로젝트 구조

```
wafer_wtms/
│
├─ cli.py                # CLI 엔트리포인트
├─ pipeline.py           # 메인 파이프라인 오케스트레이션
├─ io_utils.py           # 입·출력, 타이머, 디버그·로그 저장
├─ prefilter.py          # 전처리 필터 체인 (CLAHE, Median, Sobel 등)
├─ glcm.py               # GLCM 기반 텍스처 필터링
├─ wavelet.py            # SWT 분해, WTM 계산, DWT 시각화
├─ wtms.py               # WTMS 계산 및 interscale ratio 테스트
├─ candidate.py          # 후보점 샘플링 (μ+3σ threshold)
├─ golden_set.py         # Golden Set 필터링
├─ postprocess.py        # 연결성분 클러스터링 및 결함 분류
├─ requirements.txt      # 필수 라이브러리 의존성
├─ input_img/            # 입력 이미지 샘플
└─ results/              # 실행별 결과 (타임스탬프별 폴더)
    └─ <YYYYMMDD_HHMMSS>/
        ├─ defect_overlay.png    # 결함 오버레이 시각화
        ├─ run.txt              # 실행 로그 및 파라미터
        └─ debug_img/           # 디버그 이미지들
```

---

## 2. 설치 및 환경 설정

본 프로젝트는 **Python 3.11** 환경에서 개발 및 테스트되었습니다. 가상환경 사용을 권장합니다.

### 환경 확인
프로젝트 디렉토리에 `.venv` 가상환경이 이미 구성되어 있습니다.

1.  **가상환경 활성화:**
    ```bash
    source .venv/bin/activate
    ```

2.  **Python 버전 확인:**
    ```bash
    python --version  # Python 3.11.13이어야 함
    ```

3.  **필수 패키지 설치 확인:**
    ```bash
    pip list | grep -E "(numpy|opencv-python|pywt|scipy|scikit-image|matplotlib)"
    ```

4.  **누락된 패키지가 있다면 설치:**
    ```bash
    pip install -r requirements.txt
    pip install matplotlib psutil  # 추가 패키지
    ```

### 환경 테스트
설치가 완료되면 다음 명령어로 환경을 테스트할 수 있습니다:

```bash
# 유닛 테스트 실행
pytest tests/test_glcm_integration.py -v

# CLI 도움말 확인
python cli.py --help
```

---

## 3. 사용법

### 기본 결함 검사 (기본 전처리: median → sobel)
```bash
python cli.py inspect -i input_img/sample_1_P.bmp --debug
```

### GLCM 텍스처 필터 포함 (권장)
```bash
python cli.py inspect -i input_img/sample_1_P.bmp \
    --prefilter_chain median glcm_texture sobel \
    --debug
```

### 고급 GLCM 파라미터 설정
```bash
python cli.py inspect -i input_img/sample_1_P.bmp \
    --prefilter_chain median glcm_multi_feature sobel \
    --glcm_window_size 11 \
    --glcm_features homogeneity contrast energy correlation \
    --glcm_smoothing_sigma 1.5 \
    --std_factor 3.0 \
    --debug
```

### 다양한 샘플 이미지 테스트
```bash
# 스크래치가 있는 샘플
python cli.py inspect -i input_img/FST_Scratch_1.bmp --debug

# 파티클 샘플
python cli.py inspect -i input_img/sample_1_P.bmp --debug

# Siltron 스크래치 샘플
python cli.py inspect -i input_img/Siltron_Scratch_1.bmp --debug
```

### Golden Set을 사용한 결함 검사
```bash
python cli.py inspect -i input_img/sample_1_P.bmp --gs golden_set.csv --debug
```

### Golden Set 생성
```bash
python cli.py build-gs -m reference_mask.png -o golden_set.csv
```

실행 시 `results/<YYYYMMDD_HHMMSS>/` 폴더가 자동 생성되며 다음 결과물을 포함합니다:

* **`defect_overlay.png`** — 결함 오버레이 시각화 (스크래치=빨강, 파티클=파랑)
* **`run.txt`** — 실행 파라미터, 타이밍 정보
* **`debug_img/`** — `--debug` 플래그 사용 시 생성되는 각 단계별 디버그 이미지

---

## 4. 알고리즘 파이프라인

| 단계 | 모듈 | 설명 |
|------|------|------|
| 1. 전처리 | `prefilter.py` | GLCM, CLAHE, Median, Sobel 등의 필터 체인 적용 |
| 2. SWT 변환 | `wavelet.py` | 2-레벨 Stationary Wavelet Transform |
| 3. WTM 계산 | `wavelet.py` | Wavelet Transform Modulus 계산 |
| 4. 후보점 샘플링 | `candidate.py` | μ+3σ threshold로 후보 픽셀 추출 |
| 5. WTMS 계산 | `wtms.py` | 각 후보점에서 WTMS 값 계산 |
| 6. Interscale Test | `wtms.py` | 멀티스케일 비율 테스트 (R < 0) |
| 7. Golden Set 필터 | `golden_set.py` | 참조 데이터 기반 거짓양성 제거 |
| 8. 후처리 | `postprocess.py` | 연결성분 분석 및 스크래치/파티클 분류 |

---

## 5. 디버그 이미지 설명

`--debug` 플래그 사용 시 생성되는 주요 디버그 이미지입니다.

| 파일명 패턴 | 내용 |
|--------|------|
| `00_dwt_quad.png` | 1-레벨 DWT 4분할 시각화 (LL/LH/HL/HH) |
| `01_prefilter.png` | 전처리 후 최종 gradient 이미지 |
| `02_wtm_lvl1.png` | Level-1 WTM 히트맵 |
| `02_wtm_lvl2.png` | Level-2 WTM 히트맵 |
| `03_raw_detect_mask.png` | Interscale 테스트 후 원시 검출 마스크 |
| `04_cleaned_detect_mask.png` | Golden Set 필터링 후 최종 마스크 |

### GLCM 전처리 사용 시 추가 디버그 이미지

GLCM 텍스처 필터를 사용하면 다음 추가 디버그 이미지들이 생성됩니다:

| 파일명 패턴 | 내용 |
|--------|------|
| `01_1_glcm_homogeneity.png` | GLCM 균질성(Homogeneity) 특성 맵 |
| `01_2_glcm_contrast.png` | GLCM 대비(Contrast) 특성 맵 |
| `01_3_glcm_energy.png` | GLCM 에너지(Energy) 특성 맵 |
| `01_4_glcm_correlation.png` | GLCM 상관관계(Correlation) 특성 맵 |
| `01_5_glcm_combined_score.png` | 결합된 GLCM 점수 맵 |
| `01_6_glcm_smoothed.png` | 스무딩된 GLCM 결과 |
| `01_7_glcm_alpha_blend.png` | 원본 이미지와 알파 블렌딩 |
| `01_8_glcm_final_result.png` | 최종 GLCM 전처리 결과 |

---

## 6. 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--prefilter_chain` | `median sobel` | 적용할 전처리 필터의 순서 |
| `--wavelet_name` | `coif6` | 웨이블릿 종류 (haar, db2, coif6 등) |
| `--std_factor` | `3.0` | 후보점 추출 임계값 (μ + k*σ) |
| `--window_hw` | `3` | WTMS 계산 윈도우 반폭 |
| `--glcm_window_size` | `11` | GLCM 계산 윈도우 크기 |
| `--glcm_features`| `homogeneity`...| 사용할 GLCM 텍스처 특징 |
| `--min_region_area` | `5` | 최소 영역 크기 (픽셀) |
| `--ecc_thr` | `0.9` | 스크래치 분류 이심률 임계값 |

---

## 7. 결함 분류

프로그램은 검출된 영역을 다음 기준으로 분류합니다:

- **스크래치 (Scratch)**: 이심률 ≥ 0.9 또는 길이 ≥ 20 픽셀
- **파티클 (Particle)**: 위 조건을 만족하지 않는 영역

결과 시각화에서 스크래치는 **빨간색**, 파티클은 **파란색**으로 표시됩니다.

---

## 8. 테스팅

### 유닛 테스트 실행
```bash
# 모든 GLCM 통합 테스트 실행
pytest tests/test_glcm_integration.py -v

# 특정 테스트만 실행
pytest tests/test_glcm_integration.py::TestFilterChainExecution::test_basic_filter_chain_with_glcm -v

# 커버리지 포함 테스트
pytest tests/ --cov=. --cov-report=html
```

### 성능 벤치마크
```bash
# 기본 파이프라인 성능 측정
python cli.py inspect -i input_img/sample_1_P.bmp

# GLCM 포함 성능 측정
python cli.py inspect -i input_img/sample_1_P.bmp --prefilter_chain median glcm_texture sobel
```

### 결과 검증
테스트 실행 후 `results/` 디렉토리에서 생성된 이미지들을 확인:
- 스크래치는 빨간색, 파티클은 파란색으로 표시
- 실행 로그에서 검출된 영역 수 및 분류 결과 확인

---

## 9. 라이선스

MIT License – 자유롭게 사용·수정·재배포 가능합니다.