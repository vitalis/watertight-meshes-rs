//! Surface close: weld, drop fins, fill boundary loops. No voxel remesh.

use crate::hole_filler::fill_holes;
use crate::mesh::Mesh;
use crate::topology::{
    analyze_topology, extract_boundary_loops, non_manifold_vertex_fans, MeshHealthReport,
};
use crate::weld::merge_by_distance;
use std::collections::{HashMap, HashSet};

/// Result of a surface close. The mesh is returned separately.
#[derive(Debug, Clone)]
pub struct CloseReport {
    pub log: Vec<String>,
    pub health: MeshHealthReport,
}

/// Repair a mesh by editing its own triangles.
///
/// Never calls the voxel remesher. Vertex positions move only when welding
/// merges points already within `weld_tolerance`.
pub fn close_surface(
    input: &Mesh,
    weld_tolerance: f64,
    fill_holes_enabled: bool,
) -> (Mesh, CloseReport) {
    let mut log = Vec::new();

    let (mut current, weld_stats) = merge_by_distance(input, weld_tolerance.max(0.0));
    log.push(format!(
        "Welded {} vertices; removed {} degenerate and {} duplicate faces",
        weld_stats.merged_vertices,
        weld_stats.degenerate_faces_removed,
        weld_stats.duplicate_faces_removed
    ));

    let removed_fins = reduce_non_manifold_edges(&mut current);
    if removed_fins > 0 {
        log.push(format!(
            "Removed {removed_fins} fin faces from edges with more than two faces"
        ));
    }

    let removed_loose = remove_lone_boundary_faces(&mut current);
    if removed_loose > 0 {
        log.push(format!(
            "Removed {removed_loose} loose triangles whose three edges were the whole boundary"
        ));
    }

    if fill_holes_enabled {
        let (filled, hole_rep) = fill_holes(&current, None);
        current = filled;
        if hole_rep.holes_closed > 0 {
            log.push(format!(
                "Filled {} boundary holes (+{} triangles)",
                hole_rep.holes_closed, hole_rep.faces_added
            ));
        }
    }

    // Hole fill can lay a new triangle onto an edge that already had two faces.
    let removed_after_fill = reduce_non_manifold_edges(&mut current);
    if removed_after_fill > 0 {
        log.push(format!(
            "Removed {removed_after_fill} faces that made an edge non-manifold again after hole fill"
        ));
    }

    let split_vertices = split_non_manifold_vertices(&mut current);
    if split_vertices > 0 {
        log.push(format!(
            "Split {split_vertices} vertices where separate fans met at one point"
        ));
    }

    let mut health = analyze_topology(&current);
    if health.is_closed && health.is_manifold && health.signed_volume < -1e-9 {
        for face in &mut current.faces {
            face.swap(1, 2);
        }
        current.compute_vertex_normals();
        log.push("Flipped face winding to make signed volume positive".to_string());
        health = analyze_topology(&current);
    }

    log.push(format!(
        "Surface close: {} vertices, {} faces. Watertight: {}, 2-manifold: {}",
        current.vertex_count(),
        current.face_count(),
        health.is_watertight,
        health.is_manifold
    ));

    (current, CloseReport { log, health })
}

/// Delete faces until every edge has at most two. Returns how many faces were removed.
fn reduce_non_manifold_edges(mesh: &mut Mesh) -> usize {
    if mesh.faces.len() < 3 {
        return 0;
    }

    let mut removed = 0;
    loop {
        let counts = edge_face_counts(&mesh.faces);
        let mut crowded: Vec<(usize, usize)> = counts
            .iter()
            .filter(|(_, faces)| faces.len() > 2)
            .map(|(edge, _)| *edge)
            .collect();
        if crowded.is_empty() {
            break;
        }
        crowded.sort_unstable();
        let edge = crowded[0];
        let incident = counts[&edge].clone();
        let victim = choose_fin(&mesh.vertices, &mesh.faces, edge, &incident, &counts);
        mesh.faces.remove(victim);
        removed += 1;
    }

    if removed > 0 {
        compact_unused_vertices(mesh);
    }
    removed
}

fn choose_fin(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    edge: (usize, usize),
    incident: &[usize],
    counts: &HashMap<(usize, usize), Vec<usize>>,
) -> usize {
    let mut best = incident[0];
    for &candidate in &incident[1..] {
        if fin_outranks(vertices, faces, edge, counts, candidate, best) {
            best = candidate;
        }
    }
    best
}

/// True when `candidate` should be deleted instead of `incumbent`.
fn fin_outranks(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    edge: (usize, usize),
    counts: &HashMap<(usize, usize), Vec<usize>>,
    candidate: usize,
    incumbent: usize,
) -> bool {
    let cand_dup = is_duplicate_on_edge(faces, candidate, counts, edge);
    let inc_dup = is_duplicate_on_edge(faces, incumbent, counts, edge);
    if cand_dup != inc_dup {
        return cand_dup;
    }

    let cand_score = boundary_score(faces[candidate], edge, counts);
    let inc_score = boundary_score(faces[incumbent], edge, counts);
    if cand_score != inc_score {
        return cand_score > inc_score;
    }

    let cand_area = triangle_area(vertices, faces[candidate]);
    let inc_area = triangle_area(vertices, faces[incumbent]);
    if (cand_area - inc_area).abs() > 1e-12 {
        return cand_area < inc_area;
    }

    candidate > incumbent
}

fn is_duplicate_on_edge(
    faces: &[[usize; 3]],
    face_index: usize,
    counts: &HashMap<(usize, usize), Vec<usize>>,
    edge: (usize, usize),
) -> bool {
    let key = face_key(faces[face_index]);
    counts[&edge]
        .iter()
        .any(|&other| other != face_index && face_key(faces[other]) == key)
}

fn boundary_score(
    face: [usize; 3],
    crowded_edge: (usize, usize),
    counts: &HashMap<(usize, usize), Vec<usize>>,
) -> usize {
    face_edges(face)
        .into_iter()
        .filter(|edge| *edge != crowded_edge)
        .filter(|edge| {
            counts
                .get(edge)
                .map(|faces| faces.len() == 1)
                .unwrap_or(false)
        })
        .count()
}

fn edge_face_counts(faces: &[[usize; 3]]) -> HashMap<(usize, usize), Vec<usize>> {
    let mut counts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (face_index, face) in faces.iter().enumerate() {
        for edge in face_edges(*face) {
            counts.entry(edge).or_default().push(face_index);
        }
    }
    counts
}

fn face_edges(face: [usize; 3]) -> [(usize, usize); 3] {
    [
        canon_edge(face[0], face[1]),
        canon_edge(face[1], face[2]),
        canon_edge(face[2], face[0]),
    ]
}

fn canon_edge(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn face_key(face: [usize; 3]) -> [usize; 3] {
    let mut key = face;
    key.sort_unstable();
    key
}

fn triangle_area(vertices: &[[f64; 3]], face: [usize; 3]) -> f64 {
    let p0 = vertices[face[0]];
    let p1 = vertices[face[1]];
    let p2 = vertices[face[2]];
    let v0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

/// Drop a triangle whose three edges are a boundary loop by themselves.
/// Filling that loop would stack a second copy on the same triangle.
fn remove_lone_boundary_faces(mesh: &mut Mesh) -> usize {
    let counts = edge_face_counts(&mesh.faces);
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for face in &mesh.faces {
        for (start, end) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            *directed.entry((start, end)).or_default() += 1;
        }
    }

    let mut remove = HashSet::new();
    for poly in extract_boundary_loops(&directed, &counts) {
        if poly.len() != 3 {
            continue;
        }
        let edges = [
            canon_edge(poly[0], poly[1]),
            canon_edge(poly[1], poly[2]),
            canon_edge(poly[2], poly[0]),
        ];
        let Some(first) = counts.get(&edges[0]) else {
            continue;
        };
        if first.len() != 1 {
            continue;
        }
        let face_index = first[0];
        let same_face = edges.iter().all(|edge| {
            counts
                .get(edge)
                .is_some_and(|faces| faces.len() == 1 && faces[0] == face_index)
        });
        if same_face {
            remove.insert(face_index);
        }
    }

    if remove.is_empty() {
        return 0;
    }

    let kept: Vec<[usize; 3]> = mesh
        .faces
        .iter()
        .enumerate()
        .filter(|(index, _)| !remove.contains(index))
        .map(|(_, face)| *face)
        .collect();
    mesh.faces = kept;
    compact_unused_vertices(mesh);
    remove.len()
}

/// Give each extra fan its own vertex at the same position. Does not move geometry.
fn split_non_manifold_vertices(mesh: &mut Mesh) -> usize {
    let counts = edge_face_counts(&mesh.faces);
    let fans = non_manifold_vertex_fans(mesh, &counts);
    if fans.is_empty() {
        return 0;
    }

    let mut added = 0;
    for (vertex, components) in fans {
        let position = mesh.vertices[vertex];
        for component in components.into_iter().skip(1) {
            let copy = mesh.vertices.len();
            mesh.vertices.push(position);
            for face_index in component {
                for slot in &mut mesh.faces[face_index] {
                    if *slot == vertex {
                        *slot = copy;
                    }
                }
            }
            added += 1;
        }
    }

    mesh.compute_vertex_normals();
    added
}

fn compact_unused_vertices(mesh: &mut Mesh) {
    let mut used = vec![false; mesh.vertices.len()];
    for face in &mesh.faces {
        for index in face {
            if *index < used.len() {
                used[*index] = true;
            }
        }
    }

    let mut remap = vec![0; mesh.vertices.len()];
    let mut vertices = Vec::new();
    for (index, is_used) in used.iter().enumerate() {
        if *is_used {
            remap[index] = vertices.len();
            vertices.push(mesh.vertices[index]);
        }
    }

    let faces = mesh
        .faces
        .iter()
        .map(|face| [remap[face[0]], remap[face[1]], remap[face[2]]])
        .collect();

    *mesh = Mesh::new(vertices, faces);
}
