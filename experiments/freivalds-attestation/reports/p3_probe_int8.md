# Adversarial probe matrix

Hardware: NVIDIA GH200 480GB  
Challenge: dim=4096 dtype=int8 comparison=bitwise  
Generated: 2026-04-29T09:16:43Z

| Scenario | Description | Verdict | max_abs_diff | prover_time_ms | reason |
|---|---|---|---|---|---|
| S0_honest | real GPU matmul | **PASS** | 0 | 30.3 | bitwise match |
| S1_cached_stale | C from a stale seed | **FAIL** | 8.88e+09 | 1.2 | bitwise mismatch: max_abs_diff=8881783259.0 |
| S2_zeros | C = 0 | **FAIL** | 5.61e+09 | 1.2 | bitwise mismatch: max_abs_diff=5614922669.0 |
| S3_random | C = unrelated bytes | **FAIL** | 2.25e+13 | 1.2 | bitwise mismatch: max_abs_diff=22481932952273.0 |
| S4_dropped_rows | first half correct, second half zeroed | **FAIL** | 5.61e+09 | 1.2 | bitwise mismatch: max_abs_diff=5614922669.0 |
| S5_quantized | B aggressively quantized then matmul | **PASS** | 0 | 1.2 | bitwise match |
| S6_stub_kernel | busy-loop kernel; C is noise | **FAIL** | 5.61e+09 | 2.9 | bitwise mismatch: max_abs_diff=5614918886.0 |
