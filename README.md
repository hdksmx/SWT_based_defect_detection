

# Wavelet‑based Scratch Detection Pipeline

고해상도 **wafer 이미지**에서 임의 방향의 **얇은 scratch**를 놓치지 않고 검출하기 위해  
2‑Level **Stationary Wavelet Transform (SWT)** + MAD thresholding을 사용하는 파이썬 파이프라인입니다.

---

## 1. 프로젝트 구조

```
wavelet_code/
│
├─ main.py               # CLI 엔트리포인트
├─ utils_io.py           # 입·출력, 타이머, 디버그·로그 저장
├─ swt_utils.py          # SWT 분해·복원, DWT 시각화(dbg)
├─ thresholding.py       # σ̂(MAD) 추정, hard/soft threshold
├─ reconstruct.py        # 밴드 조합, 역변환, magnitude·mask 생성
└─ results/
    └─ <timestamp>/      # 실행별 결과 (denoised, mask, debug_img, run_log)
```

---

## 2. 환경 요구 사항

| 패키지 | 권장 버전(≥) | 비고 |
|--------|--------------|------|
| Python | 3.10 – 3.13 | |
| NumPy | 1.26.4 | 1.x 계열 유지 권장 |
| SciPy | 1.13.1 | `median_abs_deviation` 사용 |
| PyWavelets | 1.8.0 | SWT/ISWT 지원 |
| scikit‑image | 0.25.2 | `remove_small_objects`, `skeletonize` |
| OpenCV‑Python | 4.12.0.88 | 이미지 I/O |

```bash
pip install -r requirements.txt
```

---

## 3. 명령행 사용법

```bash
python main.py <image_file>
               [--wavelet haar]      # 기본: haar
               [--level 2]           # SWT 레벨
               [-k|--k 3.0]          # Threshold 계수
               [--thr_mode soft|hard]
               [--combine keep_all|keep_hv|approx_only]
               [--debug]             # 디버그 PNG 저장
               [--save_mask]         # scratch_mask.png 저장
```

실행 시 `results/<YYYYMMDD_HHMMSS>/` 폴더가 자동 생성되며:

* **`denoised.png`** — 노이즈가 억제된 복원 이미지  
* **`scratch_mask.png`** — OR‑fused 이진 마스크(255/0, skeleton 포함)  
* **`debug_img/`** — 입력·DWT 4‑타일·레벨별 magnitude 맵 등  
* **`run_log.txt`** — 파라미터, 단계별 소요 시간 기록  

---

## 4. 주요 알고리즘 단계

| 단계 | 모듈 / 함수 | 설명 |
|------|-------------|------|
| SWT 분해 | `swt_utils.swt2_decompose` | Shift‑invariant Haar SWT (2 레벨) |
| σ̂ 추정 | `thresholding.estimate_sigma` | MAD/0.6745 → 레벨별 σ̂ |
| Threshold | `thresholding.apply_threshold` | `k·σ̂` soft/hard → 노이즈 제거 |
| 밴드 조합 | `reconstruct.combine_bands` | `keep_all / keep_hv / approx_only` |
| 역변환 | `reconstruct.reconstruct_image` | → **denoised.png** |
| Magnitude · Mask | `reconstruct.make_mask` | 레벨 OR → **scratch_mask.png** |
| 로그 & 디버그 | `utils_io.*` | PNG 저장, `run_log.txt` 작성 |

---

## 5. 디버깅 이미지 해석

| 파일 | 의미 |
|------|------|
| `00_input.png` | 전처리 전 원본(gray) |
| `00_dwt_quad.png` | 1‑Level DWT LL/LH/HL/HH 4‑타일 |
| `01_mag_lvl*.png` | SWT 레벨별 magnitude heat‑map |

---

## 6. 파라미터 튜닝 가이드

| 파라미터 | 효과 | 권장 범위 |
|----------|------|-----------|
| `--wavelet` | 필터 형태·길이 | `haar`, `sym4`, `db2` |
| `--level` | 스크래치 굵기 범위 | 2 ~ 3 |
| `-k / --k` | FP ↔ FN 트레이드오프 | 2.0 ~ 3.5 |
| `--combine` | 대각 노이즈 억제 | `keep_hv` 추천 |
| `--thr_mode` | hard: 경계 보존<br>soft: 블러 적음 | scratch 굵기에 따라 선택 |

---

## 7. 확장 가능 항목

* DTCWT 6‑direction 변환(방향성 강화)  
* Cycle‑spinning DWT 버전  
* Hough merge 기반 post‑process  
* GUI / REST API 래퍼

---

## 8. 라이선스

MIT License – 자유롭게 사용·수정·재배포 가능합니다.
