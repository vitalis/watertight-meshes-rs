//! Spatial-hash vertex welding and degenerate geometry removal.
//!
//! Provides the Rust implementation of WTiVo-FastMergeByDistance: merges vertices
//! within distance threshold ε, eliminates zero-area triangles, and removes duplicate/opposing faces.

use crate::mesh::Mesh;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics from vertex welding and face deduplication.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WeldStats {
    pub original_vertices: usize,
    pub merged_vertices: usize,
    pub original_faces: usize,
    pub degenerate_faces_removed: usize,
    pub duplicate_faces_removed: usize,
    pub final_vertices: usize,
    pub final_faces: usize,
}

impl std::fmt::Display for WeldStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Welded {} vertices ({} -> {}), removed {} degenerate and {} duplicate faces",
            self.merged_vertices,
            self.original_vertices,
            self.final_vertices,
            self.degenerate_faces_removed,
            self.duplicate_faces_removed
        )
    }
}

/// Options for spatial-hash vertex welding
#[derive(Debug, Clone, Copy)]
pub struct WeldOptions {
    pub distance: f64,
}

impl Default for WeldOptions {
    fn default() -> Self {
        Self { distance: 1e-4 }
    }
}

/// Merges vertices within `distance` threshold using 3D spatial hashing,
/// collapses duplicate/opposing surfaces, and strips unreferenced vertices.
pub fn merge_by_distance(mesh: &Mesh, distance: f64) -> (Mesh, WeldStats) {
    let mut stats = WeldStats {
        original_vertices: mesh.vertices.len(),
        original_faces: mesh.faces.len(),
        ..Default::default()
    };

    if mesh.is_empty() {
        return (mesh.clone(), stats);
    }

    let cell_size = if distance > 1e-12 { distance } else { 1e-6 };
    let inv_cell = 1.0 / cell_size;
    let dist_sq = distance * distance;

    // Spatial hash grid: (ix, iy, iz) -> Vec<vertex_index>
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut vertex_remap: Vec<usize> = Vec::with_capacity(mesh.vertices.len());
    let mut unique_vertices: Vec<[f64; 3]> = Vec::new();

    for &v in mesh.vertices.iter() {
        let cx = (v[0] * inv_cell).floor() as i64;
        let cy = (v[1] * inv_cell).floor() as i64;
        let cz = (v[2] * inv_cell).floor() as i64;

        let mut matched_idx = None;

        // Check 3x3x3 neighborhood of cells
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(candidates) = grid.get(&key) {
                        for &cand_idx in candidates {
                            let cand = unique_vertices[cand_idx];
                            let dx = v[0] - cand[0];
                            let dy = v[1] - cand[1];
                            let dz = v[2] - cand[2];
                            if dx * dx + dy * dy + dz * dz <= dist_sq {
                                matched_idx = Some(cand_idx);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }

        match matched_idx {
            Some(idx) => {
                vertex_remap.push(idx);
            }
            None => {
                let new_idx = unique_vertices.len();
                unique_vertices.push(v);
                grid.entry((cx, cy, cz)).or_default().push(new_idx);
                vertex_remap.push(new_idx);
            }
        }
    }

    stats.merged_vertices = stats.original_vertices - unique_vertices.len();

    // Remap faces and remove degenerate triangles
    let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(mesh.faces.len());
    let mut new_face_uvs = Vec::new();
    let keep_uvs = mesh.face_uvs.len() == mesh.faces.len();
    // Canonical face tracker to remove exact duplicates and doubled back-to-back faces
    let mut seen_faces: HashMap<(usize, usize, usize), bool> = HashMap::new();

    for (face_index, &[i0, i1, i2]) in mesh.faces.iter().enumerate() {
        let r0 = vertex_remap[i0];
        let r1 = vertex_remap[i1];
        let r2 = vertex_remap[i2];

        // 1. Degenerate check: 2 or more vertices collapsed to the same index
        if r0 == r1 || r1 == r2 || r0 == r2 {
            stats.degenerate_faces_removed += 1;
            continue;
        }

        // 2. Collinear / zero-area check
        let p0 = unique_vertices[r0];
        let p1 = unique_vertices[r1];
        let p2 = unique_vertices[r2];
        let v0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            v0[1] * v1[2] - v0[2] * v1[1],
            v0[2] * v1[0] - v0[0] * v1[2],
            v0[0] * v1[1] - v0[1] * v1[0],
        ];
        let area_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        if area_sq < 1e-18 {
            stats.degenerate_faces_removed += 1;
            continue;
        }

        // 3. Duplicate and doubled back-to-back face check
        // Canonical sorted key
        let mut sorted = [r0, r1, r2];
        sorted.sort_unstable();
        let key = (sorted[0], sorted[1], sorted[2]);

        if seen_faces.contains_key(&key) {
            stats.duplicate_faces_removed += 1;
            continue;
        }

        seen_faces.insert(key, true);
        new_faces.push([r0, r1, r2]);
        if keep_uvs {
            new_face_uvs.push(mesh.face_uvs[face_index]);
        }
    }

    // Prune unused vertices and compact
    let mut vertex_used = vec![false; unique_vertices.len()];
    for &[i0, i1, i2] in &new_faces {
        vertex_used[i0] = true;
        vertex_used[i1] = true;
        vertex_used[i2] = true;
    }

    let mut compacted_vertices = Vec::new();
    let mut final_remap = vec![0; unique_vertices.len()];

    for (old_i, &used) in vertex_used.iter().enumerate() {
        if used {
            final_remap[old_i] = compacted_vertices.len();
            compacted_vertices.push(unique_vertices[old_i]);
        }
    }

    for f in &mut new_faces {
        f[0] = final_remap[f[0]];
        f[1] = final_remap[f[1]];
        f[2] = final_remap[f[2]];
    }

    stats.final_vertices = compacted_vertices.len();
    stats.final_faces = new_faces.len();

    let mut welded_mesh = Mesh::new(compacted_vertices, new_faces);
    welded_mesh.face_uvs = new_face_uvs;
    welded_mesh.base_color = mesh.base_color.clone();
    welded_mesh.compute_vertex_normals();

    (welded_mesh, stats)
}

/// Removes degenerate triangles (zero-area or collapsed vertices) from a mesh.
pub fn remove_degenerate_faces(mesh: &Mesh, min_area_sq: f64) -> (Mesh, usize) {
    let mut clean_faces = Vec::with_capacity(mesh.faces.len());
    let mut removed = 0;

    for &[i0, i1, i2] in &mesh.faces {
        if i0 == i1 || i1 == i2 || i0 == i2 {
            removed += 1;
            continue;
        }

        let p0 = mesh.vertices[i0];
        let p1 = mesh.vertices[i1];
        let p2 = mesh.vertices[i2];
        let v0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            v0[1] * v1[2] - v0[2] * v1[1],
            v0[2] * v1[0] - v0[0] * v1[2],
            v0[0] * v1[1] - v0[1] * v1[0],
        ];
        let area_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        if area_sq < min_area_sq {
            removed += 1;
            continue;
        }

        clean_faces.push([i0, i1, i2]);
    }

    let mut result = Mesh::new(mesh.vertices.clone(), clean_faces);
    result.compute_vertex_normals();
    (result, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_duplicate_vertices() {
        // Two triangles that share an edge in space, but have duplicated vertices on the edge
        let vertices = vec![
            [0.0, 0.0, 0.0],     // 0
            [1.0, 0.0, 0.0],     // 1
            [0.0, 1.0, 0.0],     // 2
            [1.0, 0.0, 0.00001], // 3: duplicate of 1
            [0.0, 1.0, 0.00001], // 4: duplicate of 2
            [1.0, 1.0, 0.0],     // 5
        ];
        let faces = vec![[0, 1, 2], [3, 5, 4]];

        let mesh = Mesh::new(vertices, faces);
        assert_eq!(mesh.vertices.len(), 6);

        let (welded, stats) = merge_by_distance(&mesh, 0.001);
        assert_eq!(stats.final_vertices, 4);
        assert_eq!(stats.merged_vertices, 2);
        assert_eq!(welded.faces.len(), 2);
    }

    #[test]
    fn test_remove_degenerate_collapsed_faces() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0001], // Almost identical to 1
        ];
        let faces = vec![[0, 1, 2]];

        let mesh = Mesh::new(vertices, faces);
        let (_welded, stats) = merge_by_distance(&mesh, 0.01);

        assert_eq!(stats.degenerate_faces_removed, 1);
        assert_eq!(stats.final_faces, 0);
    }

    #[test]
    fn test_duplicate_faces_removal() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // Duplicate triangle back-to-back
        let faces = vec![[0, 1, 2], [0, 2, 1]];
        let mesh = Mesh::new(vertices, faces);
        let (welded, stats) = merge_by_distance(&mesh, 1e-4);
        assert_eq!(stats.duplicate_faces_removed, 1);
        assert_eq!(welded.faces.len(), 1);
    }

    #[test]
    fn test_weld_empty_mesh() {
        let empty = Mesh::new(Vec::new(), Vec::new());
        let (welded, stats) = merge_by_distance(&empty, 1e-4);
        assert_eq!(welded.vertex_count(), 0);
        assert_eq!(stats.final_faces, 0);
    }
}
