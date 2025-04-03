## Dot Product
This section uses dot product operation to demonstrate how to use shared memory when developing a kernel to do such operation.

```
nvcc -O3 --use_fast_math \
     -gencode=arch=compute_89,code=sm_89 \
     -gencode=arch=compute_90,code=sm_90 \
     --ptxas-options=-v,-O3 \
     -Xptxas -dlcm=ca \
     -o dot-product-fix dot-product.cu
```

### Key Changes & Explanations

#### 1. Removed `-G0`
- Debug symbols are **disabled by default** in `nvcc`
- Only use `-G` when debugging is needed (e.g., for `cuda-gdb`)

#### 2. Added `-O3` to PTXAS (`--ptxas-options=-v,-O3`)
- Ensures the PTX assembler uses maximum optimizations
- `-v` shows register usage statistics
- `-O3` enables highest optimization level

#### 3. Architecture Flags
- `-gencode=arch=compute_89,code=sm_89`: Targets RTX 4060 (Ada Lovelace architecture)
- `-gencode=arch=compute_90,code=sm_90`: Optional forward compatibility for future GPUs

#### 4. Fast Math & Caching
- `--use_fast_math`: Enables faster floating-point operations (reduces precision)
- `-Xptxas -dlcm=ca`: Configures L1 cache behavior for better shared memory performance

#### 5. Optimization Levels
- `-O3`: Maximum compiler optimizations
- Avoid `-G` in production builds for better performance

### Full Optimization Summary

| Flag | Purpose | Notes |
|------|---------|-------|
| `-O3` | Maximum compiler optimizations | Preferred for release builds |
| `--use_fast_math` | Faster floating-point operations | Trade precision for speed |
| `-gencode=arch=compute_89,code=sm_89` | Targets RTX 4060 (Ampere) | Required for architecture-specific optimizations |
| `-gencode=arch=compute_90,code=sm_90` | Forward compatibility | Optional for future GPU support |
| `--ptxas-options=-v,-O3` | Shows register usage + PTX optimizations | `-v` for stats, `-O3` for max PTX optimization |
| `-Xptxas -dlcm=ca` | Configures L1 cache behavior | "cache all" for shared memory optimization |
| (No `-G`) | Disables debug symbols | Default state for maximum performance |