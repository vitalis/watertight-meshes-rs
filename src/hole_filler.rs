//! Boundary loop detection and automated hole filling.
//!
//! Closes open boundary contours in 3D AI meshes using ear-clipping and centroid
//! fan triangulation with surface normal alignment.

use crate::mesh::Mesh;
use crate::topology::extract_boundary_loops;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics from boundary hole filling.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct HoleFillerStats {
    pub holes_detected: usize,
    pub holes_closed: usize,
    pub faces_added: usize,
    pub vertices_added: usize,
}

pub type HoleFillReport = HoleFillerStats;

impl std::fmt::Display for HoleFillerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Closed {}/{} holes (+{} faces, +{} vertices)",
            self.holes_closed, self.holes_detected, self.faces_added, self.vertices_added
        )
    }
}

/// Triangulates a planar-ish 3D polygon via ear clipping (returns empty on degenerate input).
fn triangulate_loop(vertices: &[[f64; 3]], poly: &[usize]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[poly[0], poly[1], poly[2]]];
    }

    // Newell normal (area-weighted) of the loop in its traced order.
    let mut nrm = [0.0f64; 3];
    for i in 0..n {
        let p = vertices[poly[i]];
        let q = vertices[poly[(i + 1) % n]];
        nrm[0] += (p[1] - q[1]) * (p[2] + q[2]);
        nrm[1] += (p[2] - q[2]) * (p[0] + q[0]);
        nrm[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
    if len < 1e-12 {
        return Vec::new();
    }
    nrm[0] /= len;
    nrm[1] /= len;
    nrm[2] /= len;

    // Orthonormal basis (u, v) with u x v = nrm.
    let r = if nrm[0].abs() <= nrm[1].abs() && nrm[0].abs() <= nrm[2].abs() {
        [1.0, 0.0, 0.0]
    } else if nrm[1].abs() <= nrm[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let mut u = [
        r[1] * nrm[2] - r[2] * nrm[1],
        r[2] * nrm[0] - r[0] * nrm[2],
        r[0] * nrm[1] - r[1] * nrm[0],
    ];
    let ulen = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    u[0] /= ulen;
    u[1] /= ulen;
    u[2] /= ulen;
    let v = [
        nrm[1] * u[2] - nrm[2] * u[1],
        nrm[2] * u[0] - nrm[0] * u[2],
        nrm[0] * u[1] - nrm[1] * u[0],
    ];

    // Project to 2D.
    let points: Vec<[f64; 2]> = poly
        .iter()
        .map(|&pi| {
            let p = vertices[pi];
            [
                p[0] * u[0] + p[1] * u[1] + p[2] * u[2],
                p[0] * v[0] + p[1] * v[1] + p[2] * v[2],
            ]
        })
        .collect();

    // Ear-clip a CCW polygon (u x v = nrm ensures positive 2D area).
    let mut indices: Vec<usize> = (0..n).collect();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(n - 2);
    let mut guard = 0;
    while indices.len() > 3 {
        guard += 1;
        if guard > n * n {
            return Vec::new();
        }
        let m = indices.len();
        let mut clipped = false;
        for i in 0..m {
            let ia = indices[i];
            let ib = indices[(i + 1) % m];
            let ic = indices[(i + 2) % m];
            let a = points[ia];
            let b = points[ib];
            let c = points[ic];
            // Convex (left turn) for a CCW polygon.
            let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
            if cross <= 1e-12 {
                continue;
            }
            let mut contains = false;
            for &id in &indices {
                if id == ia || id == ib || id == ic {
                    continue;
                }
                if point_in_triangle(a, b, c, points[id]) {
                    contains = true;
                    break;
                }
            }
            if !contains {
                tris.push([poly[ia], poly[ib], poly[ic]]);
                indices.remove((i + 1) % m);
                clipped = true;
                break;
            }
        }
        if !clipped {
            return Vec::new();
        }
    }
    if indices.len() == 3 {
        tris.push([poly[indices[0]], poly[indices[1]], poly[indices[2]]]);
    }

    tris
}

fn point_in_triangle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let cross_ab = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
    let cross_bc = (c[0] - b[0]) * (d[1] - b[1]) - (c[1] - b[1]) * (d[0] - b[0]);
    let cross_ca = (a[0] - c[0]) * (d[1] - c[1]) - (a[1] - c[1]) * (d[0] - c[0]);
    cross_ab >= -1e-12 && cross_bc >= -1e-12 && cross_ca >= -1e-12
}

/// Fills open boundary holes in the mesh to seal them.
///
/// If `max_hole_vertices` is specified, only holes with at most this many vertices
/// will be closed (preventing sealing intentionally open planar bases if desired).
pub fn fill_holes(mesh: &Mesh, max_hole_vertices: Option<usize>) -> (Mesh, HoleFillerStats) {
    let mut stats = HoleFillerStats::default();

    if mesh.is_empty() {
        return (mesh.clone(), stats);
    }

    // Identify directed edges and face mappings
    let mut edge_to_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut directed_edges: HashMap<(usize, usize), usize> = HashMap::new();

    for (f_idx, &[i0, i1, i2]) in mesh.faces.iter().enumerate() {
        let edges = [(i0, i1), (i1, i2), (i2, i0)];
        for &(u, v) in &edges {
            let canon = if u < v { (u, v) } else { (v, u) };
            edge_to_faces.entry(canon).or_default().push(f_idx);
            *directed_edges.entry((u, v)).or_default() += 1;
        }
    }

    let loops = extract_boundary_loops(&directed_edges, &edge_to_faces);
    stats.holes_detected = loops.len();

    let mut new_vertices = mesh.vertices.clone();
    let mut new_faces = mesh.faces.clone();

    for poly in loops {
        let n = poly.len();
        if n < 3 {
            continue;
        }

        if let Some(max_v) = max_hole_vertices {
            if n > max_v {
                continue;
            }
        }

        if n == 3 {
            // Simple triangle hole
            new_faces.push([poly[0], poly[1], poly[2]]);
            stats.faces_added += 1;
            stats.holes_closed += 1;
        } else {
            let tris = triangulate_loop(&new_vertices, &poly);
            if tris.is_empty() {
                // Fallback: centroid fan (degenerate or non-planar loop).
                let mut centroid = [0.0, 0.0, 0.0];
                for &vi in &poly {
                    let p = new_vertices[vi];
                    centroid[0] += p[0];
                    centroid[1] += p[1];
                    centroid[2] += p[2];
                }
                centroid[0] /= n as f64;
                centroid[1] /= n as f64;
                centroid[2] /= n as f64;

                let c_idx = new_vertices.len();
                new_vertices.push(centroid);
                stats.vertices_added += 1;

                // Connect centroid to each edge of boundary loop
                for i in 0..n {
                    let v0 = poly[i];
                    let v1 = poly[(i + 1) % n];
                    new_faces.push([v0, v1, c_idx]);
                    stats.faces_added += 1;
                }
            } else {
                for t in tris {
                    new_faces.push(t);
                    stats.faces_added += 1;
                }
            }

            stats.holes_closed += 1;
        }
    }

    let mut filled_mesh = Mesh::new(new_vertices, new_faces);
    filled_mesh.compute_vertex_normals();

    (filled_mesh, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::analyze_topology;

    #[test]
    fn test_fill_cylinder_caps() {
        // Open cylinder has 2 holes
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, false);
        let before_rep = analyze_topology(&cylinder);
        assert_eq!(before_rep.boundary_loop_count, 2);
        assert!(!before_rep.is_closed);

        let (filled, stats) = fill_holes(&cylinder, None);
        assert_eq!(stats.holes_detected, 2);
        assert_eq!(stats.holes_closed, 2);

        let after_rep = analyze_topology(&filled);
        assert!(after_rep.is_closed, "Filled cylinder must be closed");
        assert!(after_rep.is_watertight, "Filled cylinder must be watertight");
    }

    #[test]
    fn test_fill_already_closed_mesh() {
        let cube = Mesh::cube(1.0);
        let (filled, stats) = fill_holes(&cube, None);
        assert_eq!(stats.holes_detected, 0);
        assert_eq!(stats.holes_closed, 0);
        assert_eq!(filled.face_count(), cube.face_count());
    }

    #[test]
    fn test_fill_with_max_vertices_filter() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, false);
        // Cylinder holes have 16 vertices. If max is 8, it should skip both.
        let (_filled, stats) = fill_holes(&cylinder, Some(8));
        assert_eq!(stats.holes_detected, 2);
        assert_eq!(stats.holes_closed, 0);
    }

    #[test]
    fn test_triangulate_loop_concave() {
        // Concave "L" shape in the z = 0 plane, listed counter-clockwise (+Z).
        let vertices = vec![
            [0.0, 0.0, 0.0], // 0
            [4.0, 0.0, 0.0], // 1
            [4.0, 1.0, 0.0], // 2
            [1.0, 1.0, 0.0], // 3 (concave notch)
            [1.0, 4.0, 0.0], // 4
            [0.0, 4.0, 0.0], // 5
        ];
        let poly = vec![0, 1, 2, 3, 4, 5];
        let tris = triangulate_loop(&vertices, &poly);
        assert_eq!(tris.len(), 4, "concave hexagon triangulates into n-2 = 4 triangles");
        for t in &tris {
            let p0 = vertices[t[0]];
            let p1 = vertices[t[1]];
            let p2 = vertices[t[2]];
            let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
            let cross_z = e1[0] * e2[1] - e1[1] * e2[0];
            assert!(cross_z > 0.0, "triangle {t:?} must be CCW (+Z)");
        }
    }
}

