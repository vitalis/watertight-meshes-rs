//! Mesh topological health analysis and watertightness audit.
//!
//! Evaluates 2-manifold invariants, boundary loops, non-manifold edges/vertices,
//! Euler characteristic, signed volume, and 3D printing readiness.

use crate::mesh::Mesh;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Comprehensive report of mesh topological health and watertightness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeshHealthReport {
    pub vertex_count: usize,
    pub face_count: usize,
    pub edge_count: usize,
    pub boundary_edge_count: usize,
    pub boundary_loop_count: usize,
    pub non_manifold_edge_count: usize,
    pub non_manifold_vertex_count: usize,
    pub euler_characteristic: i64,
    pub genus: Option<i64>,
    pub surface_area: f64,
    pub signed_volume: f64,
    pub is_closed: bool,
    pub is_manifold: bool,
    pub is_watertight: bool,
    pub is_3d_printable: bool,
    pub has_inverted_normals: bool,
    pub degenerate_face_count: usize,
}

impl MeshHealthReport {
    /// Formats a clean, readable terminal diagnostic box.
    pub fn format_diagnostic(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "╭────────────────────────────────────────────────────────────────────────────╮\n",
        );
        out.push_str(
            "│ 🔍 Watertight Mesh Topological Health Report                               │\n",
        );
        out.push_str(
            "╰────────────────────────────────────────────────────────────────────────────╯\n",
        );

        let status_str = if self.is_watertight {
            "✅ 100% WATERTIGHT (Clean 2-Manifold • 3D Printable • Game Ready)"
        } else if self.is_manifold && self.is_closed {
            "⚠️ CLOSED but NON-SOLID (2-manifold, zero or negative enclosed volume)"
        } else if self.is_manifold {
            "⚠️ OPEN MESH (Manifold, but contains boundary holes/gaps)"
        } else {
            "❌ NON-MANIFOLD (Contains internal faces, doubled edges, or singularities)"
        };

        out.push_str(&format!("  Status:                   {}\n\n", status_str));
        out.push_str(&format!(
            "  • Vertices:               {}\n",
            self.vertex_count
        ));
        out.push_str(&format!(
            "  • Triangles (Faces):      {}\n",
            self.face_count
        ));
        out.push_str(&format!(
            "  • Edges:                  {}\n",
            self.edge_count
        ));
        out.push_str(&format!(
            "  • Euler Characteristic:   χ = {} (V - E + F)\n",
            self.euler_characteristic
        ));

        if let Some(g) = self.genus {
            out.push_str(&format!(
                "  • Topological Genus:      g = {} (handles/holes)\n",
                g
            ));
        }

        out.push_str(&format!(
            "  • Boundary Edges (Holes): {} (in {} loops)\n",
            self.boundary_edge_count, self.boundary_loop_count
        ));
        out.push_str(&format!(
            "  • Non-Manifold Edges:     {}\n",
            self.non_manifold_edge_count
        ));
        out.push_str(&format!(
            "  • Non-Manifold Vertices:  {}\n",
            self.non_manifold_vertex_count
        ));
        out.push_str(&format!(
            "  • Degenerate Faces:       {}\n",
            self.degenerate_face_count
        ));
        out.push_str(&format!(
            "  • Surface Area:           {:.4} units²\n",
            self.surface_area
        ));
        out.push_str(&format!(
            "  • Signed Volume:          {:.4} units³\n",
            self.signed_volume
        ));

        out.push_str("\n  Readiness Checklist:\n");
        out.push_str(&format!(
            "  [{}] Closed (zero open boundaries)\n",
            if self.is_closed { "✓" } else { "✗" }
        ));
        out.push_str(&format!(
            "  [{}] 2-Manifold (no non-manifold edges or vertices)\n",
            if self.is_manifold { "✓" } else { "✗" }
        ));
        out.push_str(&format!(
            "  [{}] Positive Enclosed Volume (solid body)\n",
            if self.signed_volume > 1e-9 {
                "✓"
            } else {
                "✗"
            }
        ));
        out.push_str(&format!(
            "  [{}] 3D Slicer / Print Ready\n",
            if self.is_3d_printable { "✓" } else { "✗" }
        ));

        out
    }
}

impl std::fmt::Display for MeshHealthReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_diagnostic())
    }
}

/// Analyzes the topology of a mesh.
pub fn analyze_topology(mesh: &Mesh) -> MeshHealthReport {
    if mesh.is_empty() {
        return MeshHealthReport {
            vertex_count: 0,
            face_count: 0,
            edge_count: 0,
            boundary_edge_count: 0,
            boundary_loop_count: 0,
            non_manifold_edge_count: 0,
            non_manifold_vertex_count: 0,
            euler_characteristic: 0,
            genus: None,
            surface_area: 0.0,
            signed_volume: 0.0,
            is_closed: false,
            is_manifold: false,
            is_watertight: false,
            is_3d_printable: false,
            has_inverted_normals: false,
            degenerate_face_count: 0,
        };
    }

    // Map: canonical undirected edge (min(u,v), max(u,v)) -> list of face indices
    let mut edge_to_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    // Directed edges for boundary loops: u -> v
    let mut directed_edges: HashMap<(usize, usize), usize> = HashMap::new();
    let mut degenerate_faces = 0;
    let mut valid_face_count = 0usize;
    let mut valid_vertices: HashSet<usize> = HashSet::new();

    for (f_idx, &[i0, i1, i2]) in mesh.faces.iter().enumerate() {
        // Degenerate triangle check: out-of-range, duplicate vertex indices, or zero area
        let out_of_range =
            i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len();
        let is_idx_degenerate = i0 == i1 || i1 == i2 || i0 == i2;
        let is_area_degenerate = if !out_of_range && !is_idx_degenerate {
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
            cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2] < 1e-14
        } else {
            false
        };

        if out_of_range || is_idx_degenerate || is_area_degenerate {
            degenerate_faces += 1;
            continue;
        }

        valid_face_count += 1;
        valid_vertices.insert(i0);
        valid_vertices.insert(i1);
        valid_vertices.insert(i2);

        let edges = [(i0, i1), (i1, i2), (i2, i0)];
        for &(u, v) in &edges {
            let canon = if u < v { (u, v) } else { (v, u) };
            edge_to_faces.entry(canon).or_default().push(f_idx);
            *directed_edges.entry((u, v)).or_default() += 1;
        }
    }

    let edge_count = edge_to_faces.len();
    let mut boundary_edges = Vec::new();
    let mut non_manifold_edges = 0;

    for (&edge, faces) in &edge_to_faces {
        match faces.len() {
            1 => boundary_edges.push(edge),
            count if count > 2 => non_manifold_edges += 1,
            _ => {}
        }
    }

    // Find and count boundary loops
    let boundary_loops = extract_boundary_loops(&directed_edges, &edge_to_faces);
    let boundary_loop_count = boundary_loops.len();

    // Non-manifold vertices:
    // Check if for each vertex, the adjacent faces around it form a single fan/umbrella
    let non_manifold_vertices = count_non_manifold_vertices(mesh, &edge_to_faces);

    // Compute surface area and signed volume
    let mut surface_area = 0.0;
    let mut signed_volume = 0.0;

    for &[i0, i1, i2] in &mesh.faces {
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let p0 = mesh.vertices[i0];
        let p1 = mesh.vertices[i1];
        let p2 = mesh.vertices[i2];

        // Triangle area = 0.5 * |v0 x v1|
        let v0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            v0[1] * v1[2] - v0[2] * v1[1],
            v0[2] * v1[0] - v0[0] * v1[2],
            v0[0] * v1[1] - v0[1] * v1[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        surface_area += area;

        // Signed volume of tetrahedron from origin = (1/6) * (p0 . (p1 x p2))
        let det = p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) - p0[1] * (p1[0] * p2[2] - p1[2] * p2[0])
            + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]);
        signed_volume += det / 6.0;
    }

    let vertex_count = mesh.vertices.len();
    let face_count = mesh.faces.len();
    // Euler characteristic is only meaningful over the valid (non-degenerate) complex.
    let euler = (valid_vertices.len() as i64) - (edge_count as i64) + (valid_face_count as i64);

    let is_closed = boundary_edges.is_empty();
    let is_manifold = non_manifold_edges == 0 && non_manifold_vertices == 0;
    let is_watertight = is_closed && is_manifold && signed_volume > 1e-9;
    let is_3d_printable = is_watertight;

    // Genus g is well-defined for closed orientable surfaces: chi = 2 - 2g => g = (2 - chi) / 2
    let genus = if is_closed && is_manifold && degenerate_faces == 0 && euler % 2 == 0 {
        Some((2 - euler) / 2)
    } else {
        None
    };

    MeshHealthReport {
        vertex_count,
        face_count,
        edge_count,
        boundary_edge_count: boundary_edges.len(),
        boundary_loop_count,
        non_manifold_edge_count: non_manifold_edges,
        non_manifold_vertex_count: non_manifold_vertices,
        euler_characteristic: euler,
        genus,
        surface_area,
        signed_volume,
        is_closed,
        is_manifold,
        is_watertight,
        is_3d_printable,
        has_inverted_normals: signed_volume < -1e-9,
        degenerate_face_count: degenerate_faces,
    }
}

/// Traces boundary edge segments into closed polygonal cycles.
pub fn extract_boundary_loops(
    directed_edges: &HashMap<(usize, usize), usize>,
    edge_to_faces: &HashMap<(usize, usize), Vec<usize>>,
) -> Vec<Vec<usize>> {
    // Map outgoing boundary edge: u -> v
    let mut next_vertex: HashMap<usize, usize> = HashMap::new();

    for &(u, v) in directed_edges.keys() {
        let canon = if u < v { (u, v) } else { (v, u) };
        if let Some(faces) = edge_to_faces.get(&canon) {
            if faces.len() == 1 {
                // In a manifold surface, a boundary edge has consistent winding
                next_vertex.insert(u, v);
            }
        }
    }

    let mut loops = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    for &start in next_vertex.keys() {
        if visited.contains(&start) {
            continue;
        }

        let mut current_loop = Vec::new();
        let mut curr = start;

        while !visited.contains(&curr) {
            visited.insert(curr);
            current_loop.push(curr);
            if let Some(&nxt) = next_vertex.get(&curr) {
                if nxt == start {
                    loops.push(current_loop);
                    break;
                }
                curr = nxt;
            } else {
                // Open path (not a closed loop)
                loops.push(current_loop);
                break;
            }
        }
    }

    loops
}

/// Counts vertices where adjacent faces form multiple disconnected fans/umbrellas.
fn count_non_manifold_vertices(
    mesh: &Mesh,
    edge_to_faces: &HashMap<(usize, usize), Vec<usize>>,
) -> usize {
    non_manifold_vertex_fans(mesh, edge_to_faces).len()
}

/// Fans at vertices where the faces do not form a single umbrella.
///
/// Each entry is `(vertex, components)`. `components[0]` stays on the original
/// vertex. Later components are the extra sheets joined only at that vertex.
pub(crate) fn non_manifold_vertex_fans(
    mesh: &Mesh,
    edge_to_faces: &HashMap<(usize, usize), Vec<usize>>,
) -> Vec<(usize, Vec<Vec<usize>>)> {
    // Map vertex -> adjacent faces
    let mut vertex_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();
    for (f_idx, &[i0, i1, i2]) in mesh.faces.iter().enumerate() {
        vertex_to_faces.entry(i0).or_default().push(f_idx);
        vertex_to_faces.entry(i1).or_default().push(f_idx);
        vertex_to_faces.entry(i2).or_default().push(f_idx);
    }

    let mut fans = Vec::new();

    for (&v, faces) in &vertex_to_faces {
        if faces.len() < 2 {
            continue;
        }

        // Check if faces sharing this vertex can be walked as a single connected umbrella
        // Build graph of faces adjacent by an edge that contains v
        let mut face_adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &f in faces {
            face_adj.entry(f).or_default();
        }

        for i in 0..faces.len() {
            for j in i + 1..faces.len() {
                let f1 = faces[i];
                let f2 = faces[j];
                // Check if f1 and f2 share an edge that contains vertex v
                let tri1 = mesh.faces[f1];
                let tri2 = mesh.faces[f2];
                let shared_v: HashSet<usize> = [tri1[0], tri1[1], tri1[2]]
                    .into_iter()
                    .filter(|x| tri2.contains(x))
                    .collect();

                if shared_v.len() >= 2 && shared_v.contains(&v) {
                    let other = *shared_v.iter().find(|&&x| x != v).unwrap();
                    let canon = if v < other { (v, other) } else { (other, v) };
                    if let Some(edge_faces) = edge_to_faces.get(&canon) {
                        if edge_faces.contains(&f1) && edge_faces.contains(&f2) {
                            face_adj.entry(f1).or_default().push(f2);
                            face_adj.entry(f2).or_default().push(f1);
                        }
                    }
                }
            }
        }

        let mut visited_faces = HashSet::new();
        let mut components = Vec::new();

        for &f in faces {
            if !visited_faces.contains(&f) {
                let mut component = Vec::new();
                let mut queue = vec![f];
                visited_faces.insert(f);

                while let Some(curr) = queue.pop() {
                    component.push(curr);
                    if let Some(neighbors) = face_adj.get(&curr) {
                        for &n in neighbors {
                            if visited_faces.insert(n) {
                                queue.push(n);
                            }
                        }
                    }
                }
                components.push(component);
            }
        }

        if components.len() > 1 {
            fans.push((v, components));
        }
    }

    fans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_is_100_percent_watertight() {
        let cube = Mesh::cube(2.0);
        let report = analyze_topology(&cube);

        assert!(report.is_closed, "Cube must be closed");
        assert!(report.is_manifold, "Cube must be manifold");
        assert!(report.is_watertight, "Cube must be watertight");
        assert_eq!(report.boundary_edge_count, 0);
        assert_eq!(report.non_manifold_edge_count, 0);
        assert_eq!(report.non_manifold_vertex_count, 0);
        assert_eq!(report.euler_characteristic, 2); // Sphere topology: chi = 2
        assert_eq!(report.genus, Some(0)); // Genus 0

        // Volume of 2x2x2 cube = 8.0
        assert!((report.signed_volume - 8.0).abs() < 1e-5);
        // Surface area = 6 * (2 * 2) = 24.0
        assert!((report.surface_area - 24.0).abs() < 1e-5);
    }

    #[test]
    fn test_open_cylinder_has_boundary_edges() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, false); // uncapped
        let report = analyze_topology(&cylinder);

        assert!(!report.is_closed, "Uncapped cylinder has open boundaries");
        assert!(!report.is_watertight, "Uncapped cylinder is not watertight");
        assert_eq!(report.boundary_loop_count, 2); // Top and bottom circles
        assert_eq!(report.boundary_edge_count, 32); // 16 top + 16 bottom
        assert!(
            report.is_manifold,
            "Uncapped cylinder is still a manifold with boundary"
        );
    }

    #[test]
    fn test_capped_cylinder_is_watertight() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, true); // capped
        let report = analyze_topology(&cylinder);

        assert!(report.is_closed, "Capped cylinder is closed");
        assert!(report.is_watertight, "Capped cylinder is watertight");
        assert_eq!(report.boundary_edge_count, 0);
        assert_eq!(report.euler_characteristic, 2);
    }

    #[test]
    fn test_t_junction_detected_as_non_manifold() {
        let tjunction = Mesh::non_manifold_t_junction();
        let report = analyze_topology(&tjunction);

        assert!(!report.is_manifold, "T-junction must fail manifold check");
        assert_eq!(
            report.non_manifold_edge_count, 1,
            "Must find exactly 1 non-manifold edge"
        );
        assert!(!report.is_watertight);
    }

    #[test]
    fn test_empty_mesh() {
        let empty = Mesh::new(Vec::new(), Vec::new());
        let report = analyze_topology(&empty);
        assert_eq!(report.vertex_count, 0);
        assert_eq!(report.face_count, 0);
        assert!(!report.is_watertight);
    }

    #[test]
    fn test_inverted_normals_detection() {
        let mut cube = Mesh::cube(1.0);
        // Reverse winding of all faces
        for f in &mut cube.faces {
            f.swap(0, 1);
        }
        let report = analyze_topology(&cube);
        assert!(report.is_closed);
        assert!(report.is_manifold);
        assert!(report.has_inverted_normals);
        assert!(report.signed_volume < 0.0);
        assert!(
            !report.is_watertight,
            "Inverted normal cube cannot be watertight"
        );
    }

    #[test]
    fn test_degenerate_triangles_detection() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], // collinear
        ];
        let faces = vec![[0, 1, 2]];
        let mesh = Mesh::new(verts, faces);
        let report = analyze_topology(&mesh);
        assert_eq!(report.degenerate_face_count, 1);
    }

    #[test]
    fn test_out_of_range_face_index_does_not_panic() {
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 99]]; // index 99 out of range
        let mesh = Mesh::new(verts, faces);
        let report = analyze_topology(&mesh);
        assert_eq!(
            report.degenerate_face_count, 1,
            "invalid face counted as degenerate"
        );
    }

    #[test]
    fn test_euler_characteristic_excludes_degenerate_faces() {
        // Three collinear vertices form one degenerate face: no valid complex.
        let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mesh = Mesh::new(verts, faces);
        let report = analyze_topology(&mesh);
        assert_eq!(report.euler_characteristic, 0);
        assert_eq!(
            report.genus, None,
            "genus is undefined for degenerate meshes"
        );
    }

    #[test]
    fn test_status_not_mislabeled_for_negative_volume_mesh() {
        let mut cube = Mesh::cube(1.0);
        for f in &mut cube.faces {
            f.swap(0, 1); // invert winding -> closed, manifold, but negative volume
        }
        let report = analyze_topology(&cube);
        assert!(report.is_closed && report.is_manifold && !report.is_watertight);
        let diag = report.format_diagnostic();
        // A closed-but-non-solid mesh must not claim to contain boundary holes.
        assert!(!diag.contains("boundary holes"));
        assert!(diag.contains("NON-SOLID"));
    }
}
