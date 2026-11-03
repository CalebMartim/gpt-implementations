# Benchmark Summary Table

| Implementation | GPU | Core SLOC | Total SLOC | Train Time (s) | Train Tok/s | Infer B=1 Tok/s | Infer B=64 Tok/s | Peak VRAM (MB) | Final Val Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_pytorch` | NVIDIA A100-SXM4-40GB | 120 | 427 | 473.00 | 173,193.7 | 256.4 | 2,375.3 | 2675.3 | 3.6288 |
| `02_triton` | NVIDIA A100-SXM4-40GB | 704 | 1012 | 474.73 | 172,562.6 | 154.0 | 2,716.9 | 2240.4 | 3.6778 |
| `03_cuda_cpp` | NVIDIA A100-SXM4-40GB | 1032 | 1401 | 493.39 | 166,033.4 | 500.0 | 2,538.5 | 5611.9 | 3.6064 |
