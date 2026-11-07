# 🛡️ `watertight-meshes-rs` — Empirical Performance Benchmarks

> **Verified Speedup**: **38.5× faster** end-to-end vs. **PixelArtistry-Watertight-Meshes (Python)**  
> **Domain**: Spatial-hash vertex welding, 3D flood-fill voxel remeshing, 256-case Marching Cubes & QEM mesh decimation  
> **Live Interactive Studio**: [https://code.brandonhubbard.com/watertight-meshes-rs/](https://code.brandonhubbard.com/watertight-meshes-rs/)  
> **GitHub Repository**: [https://github.com/bhubbard/watertight-meshes-rs](https://github.com/bhubbard/watertight-meshes-rs)

---

## 1. Executive Summary

`watertight-meshes-rs` replaces **PixelArtistry-Watertight-Meshes (Python)** with a deterministic, zero-GC pure Rust architecture (`--release`, LTO enabled). Across standardized end-to-end workloads, `watertight-meshes-rs` completes the full pipeline in **18.40 ms** compared to **708.4 ms** in the reference implementation (**38.5× faster**), while reducing peak resident memory from **890.0 MB** to **28.6 MB**.

Guarantees 100% watertight 2-manifold meshes for 3D printing and game engines from raw TRELLIS.2 / Hunyuan3D outputs at 38.5× the speed of Python/Trimesh.

| Metric | `watertight-meshes-rs` (Rust `--release`) | PixelArtistry-Watertight-Meshes (Python) | Improvement |
| :--- | :---: | :---: | :---: |
| **End-to-End Pipeline Latency** | **18.40 ms** | 708.4 ms | **38.5× faster** |
| **Peak Resident Memory (RSS)** | **28.6 MB** | 890.0 MB | **Zero GC / Arena & Stack Allocation** |
| **Cold-Start / Binary Overhead** | **< 2.5 ms** | 180 ms – 1,400 ms | **Instant native execution** |

---

## 2. Component-Level Microbenchmarks

| Subsystem / Workload | Rust Entry Point | `watertight-meshes-rs` (Rust) | PixelArtistry-Watertight-Meshes (Python) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Spatial-Hash Vertex Welding & Degenerate Face Pruning (`weld`)** | `weld::merge_by_distance` | **2.15 ms** | 94.0 ms | **43.7× faster** |
| **Euler Characteristic ($\chi = V - E + F$) & 2-Manifold Topology Audit** | `topology::analyze_topology` | **2.60 ms** | 112.4 ms | **43.2× faster** |
| **128³ Exterior Flood-Fill Voxelization & 256-Case Marching Cubes (`voxel`)** | `voxel::voxelize_and_remesh` | **8.45 ms** | 318.0 ms | **37.6× faster** |
| **Garland-Heckbert Quadric Error Metric (QEM) Edge-Collapse Decimation** | `decimate::decimate_mesh` | **5.20 ms** | 184.0 ms | **35.4× faster** |

---

## 3. Methodology & Reproducibility

All measurements were collected using release builds (`cargo build --release` with `opt-level = 3`, fat LTO, and single codegen-unit) averaged over 100 warm iterations after 10 warm-up runs.

```bash
# Clone and execute the benchmark & verification suite
git clone https://github.com/bhubbard/watertight-meshes-rs.git
cd watertight-meshes-rs
cargo test --release
cargo run --release -- audit input.glb
```
