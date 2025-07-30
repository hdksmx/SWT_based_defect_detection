
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
├─ wavelet.py            # SWT 분해, WTM 계산, DWT 시각화
├─ wtms.py               # WTMS 계산 및 interscale ratio 테스트
├─ candidate.py          # 후보점 샘플링 (μ+3σ threshold)
├─ golden_set.py         # Golden Set 필터링
├─ postprocess.py        # 연결성분 클러스터링 및 결함 분류
├─ input_img/            # 입력 이미지 샘플
└─ results/              # 실행별 결과 (타임스탬프별 폴더)
    └─ <YYYYMMDD_HHMMSS>/
        ├─ defect_overlay.png    # 결함 오버레이 시각화
        ├─ run.txt              # 실행 로그 및 파라미터
        └─ debug_img/           # 디버그 이미지들
```

---

## 2. 환경 요구 사항

| 패키지 | 권장 버전(≥) | 용도 |
|--------|--------------|------|
| Python | 3.10 – 3.13 | |
| NumPy | 1.26.4 | 배열 연산 |
| OpenCV‑Python | 4.12.0.88 | 이미지 I/O 및 전처리 |
| scikit‑image | 0.25.2 | 연결성분 분석, 형태학적 연산 |
| PyWavelets | 1.8.0 | Stationary Wavelet Transform |

```bash
pip install numpy opencv-python scikit-image pywavelets
```

---

## 3. 사용법

### 기본 결함 검사
```bash
python cli.py inspect -i input_img/sample_1_P.bmp
```

### Golden Set을 사용한 결함 검사
```bash
python cli.py inspect -i input_img/sample_1_P.bmp --gs golden_set.csv
```

### Golden Set 생성
```bash
python cli.py build-gs -m reference_mask.png -o golden_set.csv
```

### 고급 파라미터 설정
```bash
python cli.py inspect -i input_img/sample_1_P.bmp \
    --prefilter_chain clahe median sobel \
    --std_factor 2.5 \
    --window_hw 5 \
    --wavelet_name coif6
```

실행 시 `results/<YYYYMMDD_HHMMSS>/` 폴더가 자동 생성되며:

* **`defect_overlay.png`** — 결함 오버레이 시각화 (스크래치=빨강, 파티클=파랑)
* **`run.txt`** — 실행 파라미터, 타이밍 정보  
* **`debug_img/`** — 각 단계별 디버그 이미지들  

---

## 4. 알고리즘 파이프라인

| 단계 | 모듈 | 설명 |
|------|------|------|
| 1. 전처리 | `prefilter.py` | CLAHE, Median, Sobel 등의 필터 체인 적용 |
| 2. SWT 변환 | `wavelet.py` | 2-레벨 Stationary Wavelet Transform |
| 3. WTM 계산 | `wavelet.py` | Wavelet Transform Modulus 계산 |
| 4. 후보점 샘플링 | `candidate.py` | μ+3σ threshold로 후보 픽셀 추출 |
| 5. WTMS 계산 | `wtms.py` | 각 후보점에서 WTMS 값 계산 |
| 6. Interscale Test | `wtms.py` | 멀티스케일 비율 테스트 (R < 0) |
| 7. Golden Set 필터 | `golden_set.py` | 참조 데이터 기반 거짓양성 제거 |
| 8. 후처리 | `postprocess.py` | 연결성분 분석 및 스크래치/파티클 분류 |

---

## 5. 디버그 이미지 설명

| 파일명 | 내용 |
|--------|------|
| `00_dwt_quad.png` | 1-레벨 DWT 4분할 시각화 (LL/LH/HL/HH) |
| `01_prefilter.png` | 전처리 후 gradient 이미지 |
| `02_wtm_lvl1.png` | Level-1 WTM 히트맵 |
| `02_wtm_lvl2.png` | Level-2 WTM 히트맵 |
| `03_raw_detect_mask.png` | Interscale 테스트 후 원시 검출 마스크 |
| `04_cleaned_detect_mask.png` | Golden Set 필터링 후 최종 마스크 |

---

## 6. 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--wavelet_name` | `coif6` | 웨이블릿 종류 (haar, db2, coif6 등) |
| `--std_factor` | `3.0` | 후보점 추출 임계값 (μ + k*σ) |
| `--window_hw` | `3` | WTMS 계산 윈도우 반폭 |
| `--max_steps` | `5` | Edge search 최대 단계 |
| `--prefilter_chain` | `median sobel` | 전처리 필터 체인 |
| `--min_region_area` | `5` | 최소 영역 크기 (픽셀) |
| `--ecc_thr` | `0.9` | 스크래치 분류 이심률 임계값 |

---

## 7. 결함 분류

프로그램은 검출된 영역을 다음 기준으로 분류합니다:

- **스크래치 (Scratch)**: 이심률 ≥ 0.9 또는 길이 ≥ 20 픽셀
- **파티클 (Particle)**: 위 조건을 만족하지 않는 영역

결과 시각화에서 스크래치는 **빨간색**, 파티클은 **파란색**으로 표시됩니다.

---

## 8. 라이선스

MIT License – 자유롭게 사용·수정·재배포 가능합니다.