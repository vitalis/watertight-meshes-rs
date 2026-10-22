//! # Watertight-Meshes-RS
//!
//! High-performance Rust engine and cross-platform CLI for generating, validating,
//! voxelizing, and repairing 100% watertight 3D AI meshes.
//!
//! Fork and complete native Rust implementation of the PixelArtistry ComfyUI Watertight
//! pipeline (TRELLIS.2 / Pixal3D / MostAadTech WTiVo).

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

pub use decimate::{decimate_mesh, decimate_with_options, DecimateOptions, DecimateReport, DecimationStats};
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
            weld_stats.merged_vertices, removed_degen + weld_stats.degenerate_faces_removed
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
            log.push("Mesh is already topologically watertight and manifold; skipping voxel remesh.".into());
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
        assert!(health.is_watertight, "Repaired cylinder must be 100% watertight");
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
        assert!(health.is_watertight, "Non-manifold repair must yield watertight surface");
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
}

