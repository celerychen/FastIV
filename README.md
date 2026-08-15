# FastIV

Fast image and vision. FastIV aims at the optimal implementation of classic
algorithms on heterogeneous chips (x86 AVX2, ARM NEON, scalar fallback) across:

- Speech and signal processing
- Image and computer vision
- Neural networks, machine learning and large models
- Financial time-series forecasting

## Build

Windows (C23-capable GCC or MSVC):

```
build\build.bat              # default: GCC
build\build.bat msvc         # MSVC
```

macOS / Linux:

```
make -C build          # build all tests (AVX2 on x86, NEON on ARM)
make -C build run      # build and run
make -C build clean
```

## Run

The build produces one standalone test binary per feature in `build/`:

- `test_darray` — dynamic array (`fiv_darray`, std::vector-style)
- `test_ctensor` — N-D tensor and binary ops
- `test_mat_transpose`, `test_mat_vec`, `test_mat_mul` — matrix ops

Each prints `PASS=n FAIL=0` on success and exits non-zero on failure.

## License

GPL v3. See [License/LICENSE](License/LICENSE). Copyright (C) 2026 Celery Chen.
