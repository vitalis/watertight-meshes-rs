//! Watertight Meshes CLI binary
//!
//! High-performance engine and cross-platform CLI for generating, validating,
//! voxelizing, and repairing 100% watertight 3D AI meshes.

use clap::{Args, Parser, Subcommand};
use colored::*;
use std::path::{Path, PathBuf};
use std::time::Instant;
use watertight::{
    analyze_topology, audit_comfyui, auto_repair, decimate_with_options, detect_comfyui_dir,
    detect_hardware, export_all_workflows, get_all_workflows, install_node_packs,
    install_workflows, voxelize_and_remesh, DecimateOptions, Mesh, RepairConfig,
    VoxelRemeshOptions,
};

#[derive(Parser)]
#[command(
    name = "watertight",
    version,
    about = "High-performance Rust engine and cross-platform CLI for 100% watertight 3D meshes",
    long_about = "A native Rust implementation and cross-platform CLI for fixing AI-generated 3D meshes (TRELLIS.2, Pixal3D).\n\
                  Includes spatial-hash vertex welding, boundary hole filling, native WTiVo voxelized Marching Cubes remeshing,\n\
                  QEM decimation, hardware environment diagnostics, and ComfyUI workflow management."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform full topological & manifold diagnostic audit on a 3D mesh
    Check(CheckArgs),

    /// Run the automated 1-click mesh repair pipeline
    Repair(RepairArgs),

    /// Voxelize and extract a 100% watertight 2-manifold surface via Marching Cubes (native WTiVo)
    Remesh(RemeshArgs),

    /// Reduce polycount using Garland-Heckbert Quadric Error Metric (QEM) edge collapse
    Decimate(DecimateArgs),

    /// Diagnose system hardware (GPU VRAM, Apple Metal, CUDA), Blender, and ComfyUI setup
    Doctor(DoctorArgs),

    /// Install ComfyUI nodes, models, and workflows (replaces Windows batch script)
    Install(InstallArgs),

    /// Manage embedded authentic PixelArtistry ComfyUI workflows
    Workflows(WorkflowsArgs),
}

#[derive(Args)]
struct CheckArgs {
    /// Path to input 3D mesh file (.obj, .stl, .ply)
    #[arg(value_name = "INPUT")]
    input: PathBuf,
}

#[derive(Args)]
struct RepairArgs {
    /// Path to input 3D mesh file (.obj, .stl, .ply)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Path to output 3D mesh file (.obj, .stl, .ply)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Spatial-hash vertex welding distance tolerance
    #[arg(long, default_value = "0.0001")]
    weld_dist: f32,

    /// Skip boundary hole filling
    #[arg(long)]
    no_fill_holes: bool,

    /// Skip voxel remeshing
    #[arg(long)]
    no_remesh: bool,

    /// Resolution for voxel remeshing (e.g. 64, 128, 256)
    #[arg(long, default_value = "128")]
    res: usize,

    /// Target face count for decimation (optional)
    #[arg(long)]
    target_faces: Option<usize>,
}

#[derive(Args)]
struct RemeshArgs {
    /// Path to input 3D mesh file (.obj, .stl, .ply)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Path to output 3D mesh file (.obj, .stl, .ply)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Resolution of the voxel bounding grid (default: 128)
    #[arg(short, long, default_value = "128")]
    resolution: usize,

    /// Grid padding percentage around the bounding box
    #[arg(long, default_value = "3")]
    padding: usize,

    /// Do not close internal cavities/floating shells
    #[arg(long)]
    keep_cavities: bool,
}

#[derive(Args)]
struct DecimateArgs {
    /// Path to input 3D mesh file (.obj, .stl, .ply)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Path to output 3D mesh file (.obj, .stl, .ply)
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    /// Target number of triangular faces
    #[arg(short, long)]
    faces: usize,

    /// Maximum quadric error tolerance
    #[arg(long, default_value = "1.0")]
    max_error: f32,
}

#[derive(Args)]
struct DoctorArgs {
    /// Optional ComfyUI root directory to audit
    #[arg(long, value_name = "PATH")]
    comfy_dir: Option<PathBuf>,
}

#[derive(Args)]
struct InstallArgs {
    /// Path to your ComfyUI installation folder
    #[arg(long, value_name = "PATH")]
    comfy_dir: Option<PathBuf>,

    /// Use newest GitHub main branches instead of tested pinned commits
    #[arg(long)]
    latest: bool,

    /// Also download required Hugging Face model checkpoints
    #[arg(long)]
    models: bool,
}

#[derive(Args)]
struct WorkflowsArgs {
    #[command(subcommand)]
    sub: WorkflowsSubcommand,
}

#[derive(Subcommand)]
enum WorkflowsSubcommand {
    /// List all embedded PixelArtistry ComfyUI workflows
    List,
    /// Export all workflows to a target directory
    Export {
        /// Destination directory
        #[arg(value_name = "DIR")]
        dir: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Check(args) => run_check(&args.input)?,
        Commands::Repair(args) => run_repair(args)?,
        Commands::Remesh(args) => run_remesh(args)?,
        Commands::Decimate(args) => run_decimate(args)?,
        Commands::Doctor(args) => run_doctor(args.comfy_dir)?,
        Commands::Install(args) => run_install(args)?,
        Commands::Workflows(args) => run_workflows(args)?,
    }

    Ok(())
}

fn run_check(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Loading mesh: {}", "==>".bold().green(), path.display());
    let start = Instant::now();
    let mesh = Mesh::load(path)?;
    println!(
        "    Loaded in {:.2}ms ({} vertices, {} faces)",
        start.elapsed().as_secs_f64() * 1000.0,
        mesh.vertex_count(),
        mesh.face_count()
    );

    let report = analyze_topology(&mesh);
    println!("\n{}", report);
    Ok(())
}

fn run_repair(args: RepairArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Loading mesh: {}", "==>".bold().green(), args.input.display());
    let start = Instant::now();
    let mesh = Mesh::load(&args.input)?;

    let config = RepairConfig {
        weld_tolerance: args.weld_dist as f64,
        fill_holes: !args.no_fill_holes,
        voxel_remesh: !args.no_remesh,
        voxel_resolution: args.res,
        target_faces: args.target_faces,
    };

    println!("{} Running automated repair pipeline...", "==>".bold().green());
    let (repaired, logs) = auto_repair(&mesh, &config)?;

    for log in logs {
        println!("  {} {}", "•".cyan(), log);
    }

    println!("{} Saving repaired mesh to: {}", "==>".bold().green(), args.output.display());
    repaired.save(&args.output)?;
    println!("{} Completed in {:.2}s", "✓".bold().green(), start.elapsed().as_secs_f64());
    Ok(())
}

fn run_remesh(args: RemeshArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Loading mesh: {}", "==>".bold().green(), args.input.display());
    let start = Instant::now();
    let mesh = Mesh::load(&args.input)?;

    println!(
        "{} Voxelizing (resolution: {}, padding: {})...",
        "==>".bold().green(),
        args.resolution,
        args.padding
    );

    let opts = VoxelRemeshOptions {
        resolution: args.resolution,
        padding: args.padding as f64 * 0.01,
        iso_level: 0.5,
        keep_cavities: args.keep_cavities,
    };

    let remeshed = voxelize_and_remesh(&mesh, &opts)?;
    println!(
        "    Output: {} vertices, {} faces (Marching Cubes)",
        remeshed.vertex_count(),
        remeshed.face_count()
    );

    println!("{} Saving to {}", "==>".bold().green(), args.output.display());
    remeshed.save(&args.output)?;
    println!("{} Remesh done in {:.2}s", "✓".bold().green(), start.elapsed().as_secs_f64());
    Ok(())
}

fn run_decimate(args: DecimateArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Loading mesh: {}", "==>".bold().green(), args.input.display());
    let start = Instant::now();
    let mesh = Mesh::load(&args.input)?;

    println!(
        "{} Decimating from {} to {} faces (QEM)...",
        "==>".bold().green(),
        mesh.face_count(),
        args.faces
    );

    let opts = DecimateOptions {
        target_faces: args.faces,
        max_error: args.max_error as f64,
        preserve_boundary: true,
    };
    let (decimated, report) = decimate_with_options(&mesh, &opts);
    println!(
        "    Decimated: {} vertices, {} faces (reduced by {:.1}%)",
        decimated.vertex_count(),
        decimated.face_count(),
        report.reduction_percentage
    );

    println!("{} Saving to {}", "==>".bold().green(), args.output.display());
    decimated.save(&args.output)?;
    println!("{} Decimation done in {:.2}s", "✓".bold().green(), start.elapsed().as_secs_f64());
    Ok(())
}

fn run_doctor(hint: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let hw = detect_hardware();
    println!("{}", hw);

    println!("\n{}", "--- ComfyUI Environment Audit ---".bold().cyan());
    if let Some(comfy_dir) = detect_comfyui_dir(hint.as_deref()) {
        println!("  ComfyUI Directory: {}", comfy_dir.display().to_string().bold());
        let audit = audit_comfyui(&comfy_dir);
        println!(
            "  Node Packs:        {}/{} installed",
            audit.nodes_present.to_string().green(),
            audit.nodes_present + audit.nodes_missing
        );
        println!(
            "  Model Weights:     {}/{} downloaded",
            audit.models_present.to_string().green(),
            audit.models_present + audit.models_missing
        );
        println!(
            "  Workflows Sidebar: {}",
            if audit.workflows_installed {
                "Installed".green()
            } else {
                "Not Installed (run 'watertight install')".yellow()
            }
        );
    } else {
        println!("  ComfyUI not detected in standard paths. Specify with '--comfy-dir <PATH>'.");
    }

    Ok(())
}

fn run_install(args: InstallArgs) -> Result<(), Box<dyn std::error::Error>> {
    let comfy_dir = match detect_comfyui_dir(args.comfy_dir.as_deref()) {
        Some(dir) => dir,
        None => {
            eprintln!(
                "{} Could not locate ComfyUI root directory. Please pass '--comfy-dir /path/to/ComfyUI'",
                "Error:".bold().red()
            );
            std::process::exit(1);
        }
    };

    println!(
        "{} Target ComfyUI directory: {}",
        "==>".bold().green(),
        comfy_dir.display()
    );

    // 1. Install Node packs
    install_node_packs(&comfy_dir, !args.latest)?;

    // 2. Install Workflows
    install_workflows(&comfy_dir)?;

    // 3. Models if requested
    if args.models {
        println!("\n{} Checking / downloading model files...", "==>".bold().green());
        let models_dir = comfy_dir.join("models");
        for model in watertight::installer::MODEL_FILES {
            if let Err(e) = watertight::installer::download_model(&models_dir, model) {
                eprintln!("  {} Failed to download {}: {}", "✗".red(), model.filename, e);
            }
        }
    } else {
        println!(
            "\n{} Model checkpoints not downloaded. Pass '--models' to fetch ~6GB of safetensors.",
            "Info:".bold().yellow()
        );
    }

    println!("\n{} Setup completed successfully!", "✓".bold().green());
    Ok(())
}

fn run_workflows(args: WorkflowsArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.sub {
        WorkflowsSubcommand::List => {
            println!("{}", "Embedded Authentic PixelArtistry Workflows:".bold().green());
            for wf in get_all_workflows() {
                println!(
                    "\n  {} {}\n     Description: {}\n     Target:      {}\n     JSON Size:   {} KB",
                    "•".cyan(),
                    wf.name.bold(),
                    wf.description,
                    wf.target_use_case.dimmed(),
                    wf.json_content.len() / 1024
                );
            }
        }
        WorkflowsSubcommand::Export { dir } => {
            println!("{} Exporting workflows to {}", "==>".bold().green(), dir.display());
            let count = export_all_workflows(&dir)?;
            println!("{} Successfully exported {} workflows.", "✓".green(), count);
        }
    }
    Ok(())
}
