# Matrix Multiplication Optimization

This project explores performance optimization techniques for dense matrix multiplication in C++. The implementations progressively improve performance through memory layout optimizations, cache-aware algorithms, SIMD vectorization, and multithreading.

## Optimization Techniques

- Loop reordering
- Contiguous memory layout
- Compiler optimization flags
- Cache blocking
- AVX2 SIMD vectorization
- Aligned memory access
- Block packing
- OpenMP multithreading

## Benchmark Results

| Implementation | Runtime (ms) |
|---|---:|
| Baseline (ijk) | 104021 |
| Loop reordered (ikj) | 42817 |
| Contiguous arrays | 22863 |
| Compiler optimizations (-O2) | 5325 |
| Cache blocking | 2939 |
| AVX2 vectorization | 1662 |
| Aligned loads/stores | 1559 |
| Block packing | 882 |
| OpenMP multithreading | 278 |
| NumPy (for comparison) | 263 |

## Project Structure

The project is divided into three stages:

- **P1** – Baseline matrix multiplication using a straightforward `ijk` loop order
- **P2** – Single-threaded optimizations, including loop reordering, contiguous memory layout, cache blocking, AVX2 vectorization, aligned memory access, and block packing
- **P3** – Multithreaded implementation using OpenMP, building on the optimized single-threaded version

## Technologies

- C++
- AVX2 intrinsics
- OpenMP

## Notes

The repository contains multiple implementations used throughout the optimization process. Matrix size, block size, thread count and active implementation can be configured directly in the source files.
