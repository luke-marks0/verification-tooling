# Adversarial probe matrix

Hardware: NVIDIA GH200 480GB  
Challenge: dim=4096 dtype=bf16 comparison=tolerance  
Generated: 2026-04-29T09:16:36Z

| Scenario | Description | Verdict | max_abs_diff | prover_time_ms | reason |
|---|---|---|---|---|---|
| S0_honest | real GPU matmul | **PASS** | 11.5 | 113.9 | tolerance match: diff=11.452417731285095 <= 601.701218295097 |
| S1_cached_stale | C from a stale seed | **FAIL** | 9.5e+03 | 2.9 | tolerance exceeded: diff=9495.089819967747 > 661.99022440910 |
| S2_zeros | C = 0 | **FAIL** | 6.01e+03 | 2.9 | tolerance exceeded: diff=6014.662303507328 > 0.1 |
| S3_random | C = unrelated bytes | **FAIL** | 5.99e+03 | 2.9 | tolerance exceeded: diff=5990.398524820805 > 14.958311462402 |
| S4_dropped_rows | first half correct, second half zeroed | **FAIL** | 6.01e+03 | 2.9 | tolerance exceeded: diff=6014.662303507328 > 597.31360087394 |
| S5_quantized | B aggressively quantized then matmul | **PASS** | 328 | 2.9 | tolerance match: diff=328.3678662776947 <= 625.9006784439088 |
| S6_stub_kernel | busy-loop kernel; C is noise | **FAIL** | 6.02e+03 | 0.7 | tolerance exceeded: diff=6015.243246529251 > 18.236300145834 |
