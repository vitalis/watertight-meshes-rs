# 🧊 Watertight-Meshes-RS

[![Crates.io](https://img.shields.io/badge/crates.io-v0.1.0-orange.svg)](https://crates.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-brightgreen.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/bhubbard/watertight-meshes-rs)

> **High-performance Rust engine and cross-platform CLI for generating, validating, voxelizing, and repairing 100% watertight 3D AI meshes.**

A native Rust fork and expansion of [PixelArtistry's Watertight Meshes](https://github.com/pixelartistry/PixelArtistry-Watertight-Meshes) pipeline (announced by [@philippsieben](https://x.com/philippsieben)).

---

## ⚡ The Problem: Why Raw 3D AI Meshes Break

State-of-the-art generative 3D models—including **Microsoft TRELLIS.2**, **Tencent Pixal3D**, and **Hunyuan3D**—revolutionize concept art, but their raw mesh outputs are notoriously flawed:

1. **Non-Manifold Geometry:** Internal T-junctions, singular vertices, and shared edges between $>2$ faces that cause 3D slicers (Bambu Studio, PrusaSlicer, Cura) to crash or generate empty toolpaths.
2. **Open Boundary Holes:** Incomplete surfaces and gaps that destroy physical volume calculations.
3. **Internal Floating Shells:** Trellis2 generates concentric interior cavities that trap resin/filament or waste render polygons.
4. **Platform Lock-In:** Upstream pipelines rely on an 87 KB Windows-only `.bat` installer script with rigid Python 3.12 / CUDA 12.8 wheel requirements that fail on macOS (Apple Silicon) or Linux.

`watertight-meshes-rs` solves this with a **pure native Rust engine** and an intuitive CLI that guarantees mathematically sound, **100% watertight 2-manifold surfaces** on any operating system.

---

## 🚀 Features

- **🛡️ 100% Watertight Guarantee:** Implements native voxelization + 3D flood-fill + 256-case Marching Cubes isosurface extraction (native Rust implementation of MostAadTech WTiVo).
- **🔍 Topological Health Audit:** Computes Euler characteristic ($\chi = V - E + F$), topological genus, boundary loops, non-manifold edges, non-manifold vertices, signed volume, and surface area.
- **⚡ Spatial-Hash Vertex Welding:** $O(N)$ 3D spatial grid distance welding to collapse doubled vertices and strip degenerate/collinear faces.
- **🕳️ Boundary Hole Filler:** Detects directed boundary loops and seals them with surface-normal-aligned fan triangulation.
- **📉 Garland-Heckbert QEM Decimation:** Quadric Error Metric edge collapse to reduce high-poly meshes down to target polycounts (LOD generation) with normal-flip prevention.
- **🍎 Truly Cross-Platform:** Runs natively on macOS (Apple Silicon Metal & Unified Memory), Linux, and Windows with zero native C++/CUDA compiler dependencies for core mesh processing.
- **🩺 System & Environment Doctor:** Probes GPU VRAM, Apple Metal, Blender headless paths, and ComfyUI installations.
- **📦 Embedded Authentic Workflows:** Ships with the 4 original PixelArtistry ComfyUI workflows embedded directly in the binary.
- **💾 Universal 3D Formats:** Direct import/export for ASCII OBJ, Binary & ASCII STL, ASCII PLY, and binary glTF (GLB).

---

## 📦 Installation

### From Source (Cargo)

```bash
git clone https://github.com/bhubbard/watertight-meshes-rs.git
cd watertight-meshes-rs
cargo build --release
```

The compiled binary will be located at `target/release/watertight`.

To install it system-wide:
```bash
cargo install --path .
```

---

## 🛠️ CLI Quickstart

### 1. Audit Mesh Topological Health
Inspect any 3D AI mesh before 3D printing or importing into game engines (Unreal, Unity, Godot):

```bash
watertight check raw_trellis_mesh.obj
```

**Example Output:**
```text
╭────────────────────────────────────────────────────────────────────────────╮
│ 🔍 Watertight Mesh Topological Health Report                               │
╰────────────────────────────────────────────────────────────────────────────╯
  Status:                   ✅ 100% WATERTIGHT (Clean 2-Manifold • 3D Printable • Game Ready)

  • Vertices:               18024
  • Triangles (Faces):      36048
  • Edges:                  54072
  • Euler Characteristic:   χ = 2 (V - E + F)
  • Topological Genus:      g = 0 (handles/holes)
  • Boundary Edges (Holes): 0 (in 0 loops)
  • Non-Manifold Edges:     0
  • Non-Manifold Vertices:  0
  • Degenerate Faces:       0
  • Surface Area:           32.2159 units²
  • Signed Volume:          2.2285 units³

  Readiness Checklist:
  [✓] Closed (zero open boundaries)
  [✓] 2-Manifold (no non-manifold edges or vertices)
  [✓] Positive Enclosed Volume (solid body)
  [✓] 3D Slicer / Print Ready
```

---

### 2. 1-Click Automated Repair Pipeline
Runs vertex welding, boundary hole closing, and (if non-manifold) native WTiVo voxel remeshing:

```bash
watertight repair input_mesh.obj -o repaired_mesh.stl --res 128
```

Options:
- `--weld-dist <FLOAT>`: Spatial-hash distance tolerance (default: `0.0001`)
- `--res <INT>`: Voxel bounding grid resolution (default: `128`)
- `--target-faces <INT>`: Decimate output to target polygon count (optional)
- `--no-fill-holes`: Skip topological boundary filling
- `--no-remesh`: Close on the original surface (weld, fin drop, hole fill) and skip Marching Cubes. Writes the mesh either way. Exits non-zero when it is still not watertight. Ignores `--target-faces`. When a welded vertex has more than one UV, a PNG base color is sampled into vertex colors and the UV map is left out of the GLB.

---

### 3. Voxelized Watertight Remeshing (Native WTiVo)
Converts multi-shelled, non-manifold geometry into a single, closed watertight shell:

```bash
watertight remesh messy_ai_asset.ply -o solid_print.obj --resolution 128
```

---

### 4. QEM Mesh Decimation (LOD Tailoring)
Simplify high-poly voxelized assets to lightweight game-ready meshes:

```bash
watertight decimate highpoly.obj -o game_ready_lod0.obj --faces 10000
```

---

### 5. Hardware Doctor
Inspect your system for 3D AI acceleration, Blender integration, and ComfyUI setups:

```bash
watertight doctor
```

Outputs GPU capability (NVIDIA CUDA VRAM or Apple Silicon Metal), recommended WTiVo resolution scaling (1024, 1536, or 2048), Python version, Blender headless binary, and ComfyUI node health.

---

### 6. Export or Deploy Authentic ComfyUI Workflows
List and export the 4 embedded authentic PixelArtistry ComfyUI workflows:

```bash
# List workflows
watertight workflows list

# Export all JSON workflows to your ComfyUI workflows folder
watertight workflows export ~/ComfyUI/user/default/workflows/PixelArtistry
```

Workflows included:
1. `PixelArtistry_01_Image_to_Watertight_Mesh.json`: Image → Watertight 3D Mesh (1536 res, 3D printing ready)
2. `PixelArtistry_01b_Image_to_Watertight_Mesh_2K.json`: 2K high-detail workflow for 16GB+ VRAM (RTX 4090 / 5080 / Apple Silicon)
3. `PixelArtistry_02_Image_to_GameReady_Asset.json`: High-poly to low-poly baked game-ready LOD pipeline
4. `PixelArtistry_03_Your_Mesh_to_GameReady_Asset.json`: Retopologize and bake your own imported AI meshes

---

### 7. ComfyUI Custom Node Installer
A cross-platform replacement for `watertightMeshes_win_installer.bat`:

```bash
watertight install --comfy-dir /path/to/ComfyUI
```

Clones the 9 pinned MostAadTech node packs (`WTiVo`, `ComfyUI-Mesh-Quad-Reconstruct`, `ComfyUI-CuMesh-Decimate`, `LODTailor`, etc.) and sets up the workflows. Pass `--models` to also fetch Hugging Face model weights.

---

## 💻 Rust Library API

`watertight-meshes-rs` is also designed to be embedded directly into Rust graphics engines and servers:

```rust
use watertight::{
    analyze_topology, auto_repair, decimate_mesh, voxelize_and_remesh,
    Mesh, RepairConfig, VoxelRemeshOptions,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load any OBJ, STL, or PLY file
    let mesh = Mesh::load(Path::new("model.obj"))?;

    // 2. Audit topological invariants
    let report = analyze_topology(&mesh);
    println!("Watertight: {}, Genus: {:?}", report.is_watertight, report.genus);

    // 3. Automated 1-click repair
    let config = RepairConfig {
        voxel_resolution: 128,
        target_faces: Some(15_000),
        ..Default::default()
    };
    let (repaired, logs) = auto_repair(&mesh, &config)?;

    // 4. Save clean 3D-printable model
    repaired.save(Path::new("repaired.stl"))?;
    Ok(())
}
```

---

## 🔬 Mathematical Foundations

| Invariant | Equation / Criterion | Meaning in `watertight-meshes-rs` |
|---|---|---|
| **Euler Characteristic** | $\chi = V - E + F = 2 - 2g$ | Evaluates topological genus $g$ (handles/tunnels). A sphere/solid cube has $\chi = 2$. |
| **Boundary Criterion** | $\partial M = \emptyset$ (Boundary Edges = 0) | Every edge is shared by exactly 2 faces. Open holes have boundary edges. |
| **2-Manifold Criterion** | $\forall v \in M, \text{Link}(v) \cong S^1$ or $D^1$ | No T-junctions, non-manifold edges, or singular pinch points. |
| **Signed Volume** | $V = \frac{1}{6} \sum_{i} \mathbf{p}_{i,0} \cdot (\mathbf{p}_{i,1} \times \mathbf{p}_{i,2})$ | Validates non-negative solid displacement and outward surface normal winding. |

---

## 🤝 Upstream Credits & Acknowledgments

- **Philipp Sieben ([@philippsieben](https://x.com/philippsieben) / PixelArtistry):** Creator of the original ComfyUI Watertight workflows and curation of the 3D pipeline ([GitHub](https://github.com/pixelartistry/PixelArtistry-Watertight-Meshes)).
- **MostAadTech ([Mstafa-awad](https://github.com/Mstafa-awad)):** Author of the custom nodes: `WTiVo`, `Mesh-Quad-Reconstruct`, `CuMesh-Decimate`, `LODTailor`, `LODsmith`.
- **Microsoft TRELLIS.2 Team & Tencent Pixal3D:** Groundbreaking open-weight generative 3D diffusion models.
- **Paul Bourke:** Reference implementation for Marching Cubes 3D surface polygonization.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
