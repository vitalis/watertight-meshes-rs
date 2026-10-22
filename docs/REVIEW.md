# Mesh Pipeline Correctness Review

- **Date:** 2026-10-23
- **Scope:** `voxel.rs`, `topology.rs`, `weld.rs`, `mesh.rs`, `decimate.rs`, `hole_filler.rs`, `env.rs`, `installer.rs`, `workflows.rs`, `main.rs`
- **Verdict:** the core Marching Cubes machinery and edge-welding key in `voxel.rs` are sound, but the load path and topology report — the two things users touch first — were not robust. The P0/P1/P2 defects below were fixed in commit `0b69e70`.

---

## Summary

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | P1 | Malformed OBJ/PLY face indices panic at load | ✅ fixed |
| 2 | P1 | Euler characteristic / genus wrong with degenerate faces | ✅ fixed |
| 3 | P1 | Voxel grid OOM at doctor-recommended resolutions | ✅ fixed |
| 4 | P1 | Decimator "normal-flip prevention" was only a degeneracy check | ✅ fixed |
| 5 | P2 | Health status mislabeled closed-but-zero-volume meshes as "boundary holes" | ✅ fixed |
| 6 | P2 | PLY polygon faces silently dropped vertices past the 3rd | ✅ fixed |
| 7 | P2 | `--max-error` / `preserve_boundary` / `--keep-cavities` were dead options | ✅ fixed |
| 8 | P2 | Installer reported success even when all clones failed | ✅ fixed |
| 9 | P2 | Downloads installed truncated files as complete | ✅ fixed (length) |
| 10 | P2 | Voxelization rasterizes whole triangle AABBs (fidelity) | ✅ fixed (triangle-voxel SAT) |
| 11 | P2 | QEM priority queue is never lazily re-keyed (stale costs) | ⏳ open |
| 12 | P2 | Hole-filler centroid fan can flip/self-intersect concave holes | ✅ fixed (ear-clipping) |
| 13 | P3 | sha256 checksum verification for model downloads | ⏳ blocked (no hashes in repo) |
| 14 | P3 | `rayon` and `sha2` declared but unused (single-threaded engine) | ✅ rayon removed (sha2 retained for #13) |
| 15 | P3 | `has_inverted_normals` only detects global inversion; degenerate normals fabricate `[0,0,1]` | ⏳ open |
| 16 | P3 | `--padding` help says "voxels" but is a percentage | ✅ fixed |

---

## Details

### 1. Malformed face indices panic (P1, fixed)

`from_obj` pushed `(v_idx - 1) as usize` unchecked (negative indices wrapped via `as usize`), and `from_ply` accepted out-of-range indices. `Mesh::new` called `compute_vertex_normals` immediately, which indexed `vertices[i0]` and panicked.

**Repro (before):** `f 1 2 999` → `panicked at src/mesh.rs:143 — index out of bounds`, exit 101.

**Fix:** loaders skip faces with out-of-range/malformed indices; `compute_face_normals`/`compute_vertex_normals` bounds-check with `.get()`; `analyze_topology` skips out-of-range faces in the edge and area/volume loops.

### 2. Euler characteristic / genus with degenerate faces (P1, fixed)

Degenerate faces were skipped in edge counting but still counted in `face_count`, so `χ = V − E + F` mixed sets. A single collinear triangle reported `χ = 4` and `genus = -1`.

**Fix:** Euler is computed over the valid complex only (`valid_vertices − edges + valid_faces`); `genus` is `None` when degenerate faces are present.

### 3. Voxel OOM (P1, fixed)

The grid is `nx·ny·nz ≈ res³` and holds `occupancy` (`bool`) + `is_exterior` (`bool`) + `field` (`f32`). `env.rs` recommended up to **2048** (inherited from the GPU upstream), which is ~50 GB of CPU allocation.

**Fix:** `voxelize_and_remesh` errors before allocation when the grid exceeds ~134M cells (~800 MB).

### 4. Decimator normal-flip prevention (P1, fixed)

The collapse guard computed the post-collapse normal but only checked `len < 1e-14` (a degeneracy test); it never compared against the original normal, so flips passed.

**Fix:** `would_invert_or_degenerate` compares the post-collapse normal against the original and rejects real flips.

### 5. Status mislabel (P2, fixed)

`format_diagnostic` derived "OPEN MESH … boundary holes" from `is_manifold && !is_watertight`, but `is_watertight` is false for zero/negative volume too. A closed, negative-volume mesh reported "boundary holes" with `boundary_edge_count == 0`.

**Fix:** a distinct "CLOSED but NON-SOLID" status for `is_manifold && is_closed && !is_watertight`.

### 6. PLY n-gon truncation (P2, fixed)

A quad `4 0 1 2 3` became triangle `[0,1,2]`, dropping vertex 3.

**Fix:** `from_ply` fan-triangulates all polygon faces (matching OBJ behavior).

### 7. Dead options (P2, fixed)

`--max-error`, `--keep-cavities`, and `DecimateOptions { max_error, preserve_boundary }` were parsed/exported but never applied.

**Fix:** added `decimate_with_options` honoring `max_error` (stop threshold) and `preserve_boundary` (skip boundary-vertex collapses); `decimate_mesh` is now a backward-compatible wrapper. `--keep-cavities` is implemented (see below).

### 8. Installer false success (P2, fixed)

`install_node_packs` discarded `git clone`/`checkout` results and returned `Ok(())`, so `install` printed "Setup completed successfully!" after zero successful installs.

**Fix:** clone/checkout failures now return `Err` (verified: a failing `git` yields exit 1 with `Failed to clone …`).

### 9. Download truncation (P2, fixed)

`download_model` read `content-length` but never compared it to bytes written; a dropped connection still renamed `.part` → final.

**Fix:** `verify_download_length` rejects mismatches and deletes the partial file. (sha256 hashes remain deferred — see #13.)

### keep-cavities (new feature)

`VoxelRemeshOptions.keep_cavities` (default `false`) makes the solid field mark only surface cells, preserving enclosed interior cavities and floating shells instead of flood-filling them solid. Wired to `watertight remesh --keep-cavities`.

### 12. Hole-filler concave triangulation (P2, fixed)

The centroid fan produced correctly-oriented caps only for convex/near-planar holes; concave loops could get flipped or self-intersecting triangles. `triangulate_loop` now ear-clips the loop projected onto its Newell-normal plane (centroid fan retained as a fallback for degenerate/non-planar loops).

### 10. Voxelization fidelity (P2, fixed)

Surface rasterization marked every cell in a triangle's bounding box, so a large diagonal triangle became a thick slab. It now uses an Akenine-Möller triangle-AABB separating-axis test, marking only cells the triangle surface actually intersects — without breaking the watertightness guarantee (all 60 tests pass).

---

## Verification

- `cargo test` — 60 tests pass (55 lib + 5 CLI).
- `cargo clippy --all-targets` — clean, zero warnings.
- Manual repros: malformed-index OBJ no longer panics; degenerate mesh reports `χ = 0` (no genus); absurd `--resolution` errors cleanly instead of aborting; failing `git` makes `install` fail.

## Residual risks / deferred

- **#11 (stale QEM heap):** collapse targets/costs are not re-solved after neighboring collapses, so results are suboptimal (not incorrect).
- **#13 (checksums):** content-length verification catches truncation, but not silent corruption. Adding sha256 requires the actual hashes of the ~6 GB of model files, which are not in the repo.
- **#15:** `has_inverted_normals` only detects global inversion; degenerate normals fabricate `[0,0,1]` (the field is not displayed).
