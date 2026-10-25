//! # Watertight-Meshes-RS
//!
//! High-performance Rust engine and cross-platform CLI for generating, validating,
//! voxelizing, and repairing 100% watertight 3D AI meshes.
//!
//! Fork and complete native Rust implementation of the PixelArtistry ComfyUI Watertight
//! pipeline (TRELLIS.2 / Pixal3D / MostAadTech WTiVo).

pub mod close;
pub mod decimate;
pub mod env;
pub mod error;
pub mod hole_filler;
pub mod installer;
pub mod mesh;
pub mod topology;
pub mod voxel;
pub mod weld;
pub mod workflows;

pub use close::{close_surface, CloseReport};
pub use decimate::{
    decimate_mesh, decimate_with_options, DecimateOptions, DecimateReport, DecimationStats,
};
pub use env::{detect_hardware, EnvironmentInfo, HardwareProfile};
pub use error::{Result, WatertightError};
pub use hole_filler::{fill_holes, HoleFillReport, HoleFillerStats};
pub use installer::{
    audit_comfyui, detect_comfyui_dir, download_model, install_node_packs, install_workflows,
    CustomNodePack, InstallationReport, ModelFile, MODEL_FILES, NODE_PACKS,
};
pub use mesh::{Aabb, Face, Mesh, Point3, Vector3};
pub use topology::{analyze_topology, MeshHealthReport};
pub use voxel::{voxel_remesh, voxelize_and_remesh, VoxelRemeshOptions};
pub use weld::{merge_by_distance, remove_degenerate_faces, WeldOptions, WeldStats};
pub use workflows::{export_all_workflows, export_workflows, get_all_workflows, WorkflowMeta};

/// Configuration for the unified repair pipeline
#[derive(Debug, Clone)]
pub struct RepairConfig {
    /// Spatial-hash vertex welding distance tolerance
    pub weld_tolerance: f64,
    /// Whether to attempt topological boundary hole filling before remeshing
    pub fill_holes: bool,
    /// Whether to run the voxelized Marching Cubes remeshing (guarantees watertight 2-manifold)
    pub voxel_remesh: bool,
    /// Grid resolution along the bounding box longest axis for voxel remeshing
    pub voxel_resolution: usize,
    /// Target number of triangular faces after QEM decimation (None = keep remeshed polycount)
    pub target_faces: Option<usize>,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            weld_tolerance: 1e-4,
            fill_holes: true,
            voxel_remesh: true,
            voxel_resolution: 128,
            target_faces: Some(10_000),
        }
    }
}

/// Run the full automated repair pipeline on a 3D mesh.
///
/// Returns the repaired mesh along with an execution log of steps performed.
pub fn auto_repair(input_mesh: &Mesh, config: &RepairConfig) -> Result<(Mesh, Vec<String>)> {
    let mut log = Vec::new();
    let initial_health = analyze_topology(input_mesh);
    log.push(format!(
        "Initial status: {} vertices, {} faces. Watertight: {}, 2-Manifold: {}",
        input_mesh.vertex_count(),
        input_mesh.face_count(),
        initial_health.is_watertight,
        initial_health.is_manifold
    ));

    let mut current_mesh = input_mesh.clone();

    // 1. Vertex welding & degenerate face removal
    if config.weld_tolerance > 0.0 {
        let (welded, weld_stats) = merge_by_distance(&current_mesh, config.weld_tolerance);
        let (cleaned, removed_degen) = remove_degenerate_faces(&welded, 1e-12);
        log.push(format!(
            "Welded {} duplicate vertices; purged {} degenerate faces",
            weld_stats.merged_vertices,
            removed_degen + weld_stats.degenerate_faces_removed
        ));
        current_mesh = cleaned;
    }

    // 2. Boundary hole filling
    if config.fill_holes {
        let (patched, hole_rep) = fill_holes(&current_mesh, None);
        if hole_rep.holes_closed > 0 {
            log.push(format!(
                "Filled {} boundary holes (+{} triangles added)",
                hole_rep.holes_closed, hole_rep.faces_added
            ));
        }
        current_mesh = patched;
    }

    // Check if mesh is already watertight & manifold before voxel remeshing
    let mid_health = analyze_topology(&current_mesh);
    if config.voxel_remesh {
        if !mid_health.is_watertight || !mid_health.is_manifold {
            log.push(format!(
                "Mesh is non-manifold (holes: {}, non-manifold edges: {}). Running WTiVo voxel remesher (resolution: {})...",
                mid_health.boundary_edge_count, mid_health.non_manifold_edge_count, config.voxel_resolution
            ));
            let remesh_opts = VoxelRemeshOptions {
                resolution: config.voxel_resolution,
                padding: 0.05,
                iso_level: 0.5,
                keep_cavities: false,
            };
            current_mesh = voxelize_and_remesh(&current_mesh, &remesh_opts)?;
            log.push(format!(
                "Voxel remesh complete: produced {} vertices, {} faces (100% watertight 2-manifold guarantee)",
                current_mesh.vertex_count(),
                current_mesh.face_count()
            ));
        } else {
            log.push(
                "Mesh is already topologically watertight and manifold; skipping voxel remesh."
                    .into(),
            );
        }
    }

    // 3. QEM Decimation if requested
    if let Some(target) = config.target_faces {
        if current_mesh.face_count() > target {
            log.push(format!(
                "Decimating mesh from {} to {} faces with Quadric Error Metric...",
                current_mesh.face_count(),
                target
            ));
            let (decimated, dec_rep) = decimate_mesh(&current_mesh, target);
            log.push(format!(
                "Decimation done: {} faces remaining (reduced by {:.1}%)",
                decimated.face_count(),
                dec_rep.reduction_percentage
            ));
            current_mesh = decimated;
        }
    }

    let final_health = analyze_topology(&current_mesh);
    log.push(format!(
        "Final status: {} vertices, {} faces. Watertight: {}, 2-Manifold: {}",
        current_mesh.vertex_count(),
        current_mesh.face_count(),
        final_health.is_watertight,
        final_health.is_manifold
    ));

    Ok((current_mesh, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_repair_open_cylinder() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, false);
        let config = RepairConfig {
            weld_tolerance: 1e-4,
            fill_holes: true,
            voxel_remesh: true,
            voxel_resolution: 32,
            target_faces: None,
        };

        let (repaired, logs) = auto_repair(&cylinder, &config).expect("auto_repair failed");
        assert!(!logs.is_empty());
        let health = analyze_topology(&repaired);
        assert!(
            health.is_watertight,
            "Repaired cylinder must be 100% watertight"
        );
        assert!(health.is_manifold);
        assert_eq!(health.boundary_edge_count, 0);
    }

    #[test]
    fn test_auto_repair_non_manifold_mesh() {
        let tjunction = Mesh::non_manifold_t_junction();
        let config = RepairConfig {
            weld_tolerance: 1e-4,
            fill_holes: true,
            voxel_remesh: true,
            voxel_resolution: 24,
            target_faces: None,
        };

        let (repaired, logs) = auto_repair(&tjunction, &config).expect("auto_repair failed");
        assert!(!logs.is_empty());
        let health = analyze_topology(&repaired);
        assert!(
            health.is_watertight,
            "Non-manifold repair must yield watertight surface"
        );
        assert!(health.is_manifold);
        assert_eq!(health.non_manifold_edge_count, 0);
    }

    #[test]
    fn test_auto_repair_with_decimation() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 32, true);
        let orig_faces = cylinder.face_count();
        let config = RepairConfig {
            weld_tolerance: 1e-4,
            fill_holes: true,
            voxel_remesh: false,
            voxel_resolution: 32,
            target_faces: Some(24),
        };

        let (repaired, logs) = auto_repair(&cylinder, &config).expect("auto_repair failed");
        assert!(!logs.is_empty());
        assert!(repaired.face_count() < orig_faces);
        assert!(repaired.face_count() <= 32);
    }

    fn vertex_matches(mesh: &Mesh, v: [f64; 3]) -> bool {
        mesh.vertices.iter().any(|u| {
            (u[0] - v[0]).abs() < 1e-12
                && (u[1] - v[1]).abs() < 1e-12
                && (u[2] - v[2]).abs() < 1e-12
        })
    }

    #[test]
    fn close_open_cylinder_keeps_wall_and_seals() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, false);
        let (closed, report) = close_surface(&cylinder, 1e-4, true);
        assert!(report.health.is_watertight);
        assert!(report
            .log
            .iter()
            .all(|line| !line.to_lowercase().contains("voxel")));
        for v in &cylinder.vertices {
            assert!(vertex_matches(&closed, *v), "wall vertex moved: {v:?}");
        }
    }

    #[test]
    fn close_cube_does_not_change_geometry() {
        let cube = Mesh::cube(1.0);
        let (closed, report) = close_surface(&cube, 1e-4, true);
        assert!(report.health.is_watertight);
        assert_eq!(closed.face_count(), cube.face_count());
        assert_eq!(closed.vertices, cube.vertices);
    }

    #[test]
    fn close_inverted_cube_flips_to_positive_volume() {
        let cube = Mesh::cube(1.0);
        let flipped: Vec<[usize; 3]> = cube.faces.iter().map(|f| [f[0], f[2], f[1]]).collect();
        let inverted = Mesh::new(cube.vertices.clone(), flipped);
        assert!(analyze_topology(&inverted).signed_volume < 0.0);
        let (closed, report) = close_surface(&inverted, 1e-4, true);
        assert!(report.health.signed_volume > 0.0);
        assert!(report.health.is_watertight);
        assert_eq!(closed.vertices, inverted.vertices);
    }

    #[test]
    fn close_t_junction_clears_edge_without_making_a_solid() {
        let junction = Mesh::non_manifold_t_junction();
        let (closed, report) = close_surface(&junction, 1e-4, true);
        assert_eq!(report.health.non_manifold_edge_count, 0);
        assert!(!report.health.is_watertight);
        assert!(report
            .log
            .iter()
            .all(|line| !line.to_lowercase().contains("voxel")));
        for v in &closed.vertices {
            assert!(vertex_matches(&junction, *v), "close invented vertex {v:?}");
        }
    }

    #[test]
    fn close_open_cylinder_can_skip_hole_fill() {
        let cylinder = Mesh::cylinder(1.0, 2.0, 16, false);
        let (closed, report) = close_surface(&cylinder, 1e-4, false);
        assert!(!report.health.is_watertight);
        assert!(report.health.boundary_edge_count > 0);
        assert_eq!(closed.face_count(), cylinder.face_count());
    }

    #[test]
    fn close_pinched_cubes_splits_the_shared_vertex() {
        let mut first = Mesh::cube(1.0);
        let mut second = Mesh::cube(1.0);
        for vertex in &mut second.vertices {
            vertex[0] += 1.0;
            vertex[1] += 1.0;
            vertex[2] += 1.0;
        }
        let offset = first.vertices.len();
        first.vertices.extend(second.vertices);
        first.faces.extend(
            second
                .faces
                .into_iter()
                .map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]),
        );

        let (closed, report) = close_surface(&first, 1e-4, true);
        assert_eq!(report.health.non_manifold_vertex_count, 0);
        assert!(report.health.is_watertight);
        assert_eq!(closed.face_count(), 24);
    }

    #[test]
    fn close_drops_a_loose_triangle_instead_of_doubling_it() {
        let mut cube = Mesh::cube(1.0);
        let base = cube.vertices.len();
        cube.vertices
            .extend([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.5, 1.0, 0.0]]);
        cube.faces.push([base, base + 1, base + 2]);

        let (closed, report) = close_surface(&cube, 1e-4, true);
        assert!(report.health.is_watertight, "{:?}", report.health);
        assert_eq!(closed.face_count(), 12);
        assert_eq!(closed.vertex_count(), 8);
    }

    #[test]
    fn close_cube_with_fin_drops_fin_and_stays_watertight() {
        let cube = Mesh::cube(1.0);
        let mut finned = cube.clone();
        finned.vertices.push([0.0, -2.0, 0.0]);
        let tip = finned.vertices.len() - 1;
        finned.faces.push([0, 1, tip]);
        assert_eq!(analyze_topology(&finned).non_manifold_edge_count, 1);

        let (closed, report) = close_surface(&finned, 1e-4, true);
        assert!(report.health.is_watertight);
        assert_eq!(closed.face_count(), cube.face_count());
        assert_eq!(closed.vertices, cube.vertices);
    }
}
