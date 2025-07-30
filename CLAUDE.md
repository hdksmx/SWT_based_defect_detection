

# Wafer WTMS Scratch‑Detection  
### Developer Guide & Implementation Prompt for **CLAUDE**

This document explains how we will structure, write, and document the Python code that re‑implements the Wavelet‑Transform‑Modulus‑Sum (WTMS) scratch‑detection algorithm (now using a Stationary Wavelet Transform backend rather than a decimated DWT) described in *"A Wavelet-Based Approach in Detecting Visual Defects on Semiconductor Wafer Dies," in IEEE Transactions on Semiconductor Manufacturing, vol. 23, no. 2, pp. 284-292, May 2010, doi: 10.1109/TSM.2010.2046108*.

---

## 1. High‑Level Architecture

```
wafer_wtms/
├── cli.py                # single entry‑point
├── config.yaml           # all tunable parameters
├── pipeline.py           # orchestrates every step
├── prefilter.py          # median → sobel
├── wavelet.py            # SWT + WTM maps
├── candidate.py          # μ+3σ sampling, edge‑search
├── wtms.py               # WTMS & inter‑scale ratio
├── golden_set.py         # build/load GS mask
├── postprocess.py        # clustering & defect class
├── io_utils.py           # I/O helpers
├── utils/
│   ├── timing.py
│   └── path.py
└── tests/                # pytest unit tests
```

*Each module is self‑contained and has no side‑effects at import time.*  
Runtime order is: **prefilter → wavelet/WTM → candidate → WTMS → GS filter → postprocess**.

---

## 2. Coding Conventions

| Topic | Guideline |
|-------|-----------|
| **Python ≥3.9** | Use type hints; enable `from __future__ import annotations`. |
| **Formatting** | Black (88 cols) + isort + flake‑8; no explicit `# noqa` unless unavoidable. |
| **Docstrings** | Google style, first‑line imperative: `"Compute WTM map."` |
| **Variable names** | `img`, `wtm_map`, `coords: NDArray[int_]`; avoid one‑letter vars except indices `i,j`. |
| **Immutability** | Prefer functional style – functions should return new arrays instead of mutating in‑place, unless performance dictates otherwise. |
| **Logging** | Use Python `logging` with module‑level `logger`. Timing decorators log at INFO. |

---

## 3. Function Skeletons & Responsibilities

| Module | Key Public Functions | Contract |
|--------|----------------------|----------|
| `prefilter.py` | `median_then_sobel(img: NDArray[np.uint8], ksize: int = 3) -> NDArray[np.uint8]` | Returns edge‑enhanced 16‑bit image. |
| `wavelet.py` | `swt2(img, wavelet='coif6', level=2) -> list[CoeffTuple]`<br>`wtm(coeffs: CoeffTuple) -> NDArray[float_]` | Wrapper around `pywt.wavedec2`; `CoeffTuple = tuple[NDArray, NDArray, NDArray]`. |
| `candidate.py` | `sample_candidates(wtm_lvl1, std_factor=3.0) -> NDArray[int_]`<br>`edge_search(wtm, p: tuple[int,int], window_hw:int =3) -> list[tuple[int,int]]` | Returns pixel indices; edge_search follows 8‑dir gradient five steps. |
| `wtms.py` | `wtms_single_scale(wtm, path:list[tuple[int,int]]) -> float`<br>`interscale_ratio(w1:float, w2:float, eps=1e-9)->float` | Core WTMS maths. |
| `golden_set.py` | `build_gs(mask: NDArray[bool_], out_csv:Path)` | Save CSV of GS coords.<br>`filter_by_gs(cand: NDArray[int_], gs: set[tuple[int,int]]) -> NDArray[int_]`. |
| `postprocess.py` | `cluster_labels(mask: NDArray[bool_]) -> list[Region]` | Uses `skimage.measure.label`, `regionprops` to classify scratch vs particle. |
| `pipeline.py` | `run(img_path: Path, cfg: Config) -> ResultBundle` | Orchestrates modules, saves debug images via `utils.path.debug_path`. |

Every function must raise **`ValueError`** on invalid inputs instead of silently clipping.

---

## 4. Commenting & Inline Notes

* Keep inline comments sparing—prefer expressive variable names.  
* Any math derivation should reference the equation number in the IEEE paper, e.g. `# Eq.(4) – WTMS sum`.  
* FPGA‑specific optimisations go in block comments starting with `# [FPGA]`.

---

## 5. Debug Outputs

| Step | Filename Pattern | Purpose |
|------|-----------------|----------|
| 1    | `01_prefilter.png` | Visual sanity check of median+sobel. |
| 3    | `03_wtm_lvl1.png` | Heat‑map of local wavelet energy. |
| 6    | `06_raw_detect_mask.png` | Pixels that passed inter‑scale test (pre‑GS). |
| 8    | `08_defect_overlay.png` | Final overlay for report. |

Use `matplotlib` for colour‑mapped PNGs; save in `results/YYYYMMDD_HHMMSS/`.

---

## 6. Testing Strategy

* **Unit** – fast synthetic 64×64 images in `tests/fixtures/`; aim for ≥90 % branch coverage.
* **Regression** – real wafer tiles (`samples/`) with known mask; tolerance `dice ≥0.9`.
* **Benchmark** – `cli.py --mode bench` prints step‑wise ms and FPS.

---

## 7. How to Contribute

1. Fork branch `dev/<initials>/<topic>`.  
2. Add or update tests; run `pytest -q`.  
3. Ensure `pre‑commit run --all-files` passes.  
4. Open PR with description *“Why”, “What”, “How to test”*.

---

### Final note for Claude 🤖  
**Your objective:** implement modules so that `python -m wafer_wtms.cli --img my_wafer.bmp` runs end‑to‑end and produces `results/…/08_defect_overlay.png` identical to the reference.  

Stick to the contracts & coding style above, prefer clear logic over micro‑optimisation, and write self‑explanatory code—developers after you will port the heavy parts to Vivado HLS.

Good luck and code safely!
