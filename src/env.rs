//! System, GPU, Blender, and ComfyUI hardware environment inspection.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// System environment and 3D AI acceleration readiness report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub os: String,
    pub arch: String,
    pub gpu_name: Option<String>,
    pub gpu_vram_gb: Option<f64>,
    pub is_nvidia: bool,
    pub is_apple_silicon: bool,
    pub recommended_wtivo_res: usize,
    pub recommended_proxy_points: usize,
    pub python_version: Option<String>,
    pub blender_path: Option<PathBuf>,
    pub comfyui_path: Option<PathBuf>,
}

impl EnvironmentInfo {
    /// Dynamically probes the system hardware, GPU, Blender, and AI environment.
    pub fn probe() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        let is_apple_silicon = cfg!(all(target_os = "macos", target_arch = "aarch64"));

        // Probe NVIDIA GPU via nvidia-smi
        let (gpu_name, gpu_vram_gb, is_nvidia) = probe_nvidia_gpu();

        // Determine recommended voxel resolution. The Rust voxelizer runs on the
        // CPU and its memory guard rejects grids above ~512, so cap there.
        let (recommended_wtivo_res, recommended_proxy_points) = match gpu_vram_gb {
            Some(vram) if vram >= 15.0 => (512, 1_500_000), // 16GB+ VRAM
            Some(vram) if vram >= 7.5 => (384, 850_000),    // 8GB+ VRAM
            Some(_) => (256, 375_000),                      // lower VRAM
            None => {
                if is_apple_silicon {
                    (512, 1_500_000) // Apple Silicon Unified Memory
                } else {
                    (256, 375_000) // CPU fallback
                }
            }
        };

        let python_version = probe_python();
        let blender_path = find_blender();
        let comfyui_path = find_comfyui();

        Self {
            os,
            arch,
            gpu_name,
            gpu_vram_gb,
            is_nvidia,
            is_apple_silicon,
            recommended_wtivo_res,
            recommended_proxy_points,
            python_version,
            blender_path,
            comfyui_path,
        }
    }

    /// Formats a clean terminal diagnostic box.
    pub fn format_diagnostic(&self) -> String {
        let mut out = String::new();
        out.push_str("╭────────────────────────────────────────────────────────────────────────────╮\n");
        out.push_str("│ 🔍 Watertight 3D System & Environment Diagnostics                          │\n");
        out.push_str("╰────────────────────────────────────────────────────────────────────────────╯\n");

        out.push_str(&format!("  • Platform:               {} ({})\n", self.os, self.arch));

        if let Some(ref gpu) = self.gpu_name {
            let vram_str = self
                .gpu_vram_gb
                .map(|v| format!(" (~{:.1} GB VRAM)", v))
                .unwrap_or_default();
            out.push_str(&format!("  • GPU Acceleration:       [✓] {}{}\n", gpu, vram_str));
        } else if self.is_apple_silicon {
            out.push_str("  • GPU Acceleration:       [✓] Apple Silicon Metal (Unified Memory)\n");
        } else {
            out.push_str("  • GPU Acceleration:       [i] CPU SIMD Fallback (no discrete GPU detected)\n");
        }

        out.push_str(&format!(
            "  • WTiVo Recommended:      {} res ({} proxy points)\n",
            self.recommended_wtivo_res, self.recommended_proxy_points
        ));

        if let Some(ref py) = self.python_version {
            out.push_str(&format!("  • Python Runtime:         [✓] {}\n", py));
        } else {
            out.push_str("  • Python Runtime:         [i] None detected on PATH\n");
        }

        if let Some(ref b) = self.blender_path {
            out.push_str(&format!("  • Blender Headless:       [✓] {}\n", b.display()));
        } else {
            out.push_str("  • Blender Headless:       [i] Not found in standard directories\n");
        }

        if let Some(ref c) = self.comfyui_path {
            out.push_str(&format!("  • ComfyUI Environment:   [✓] {}\n", c.display()));
        } else {
            out.push_str("  • ComfyUI Environment:   [i] Not found in current or parent directories\n");
        }

        out
    }
}

pub type HardwareProfile = EnvironmentInfo;

impl std::fmt::Display for EnvironmentInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_diagnostic())
    }
}

/// Probes the current system and GPU hardware environment
pub fn detect_hardware() -> EnvironmentInfo {
    EnvironmentInfo::probe()
}

fn probe_nvidia_gpu() -> (Option<String>, Option<f64>, bool) {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = s.lines().next() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let name = parts[0].trim().to_string();
                    let vram_mb: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    return (Some(name), Some(vram_mb / 1024.0), true);
                } else if !parts.is_empty() {
                    return (Some(parts[0].trim().to_string()), None, true);
                }
            }
        }
    }

    (None, None, false)
}

fn probe_python() -> Option<String> {
    for cmd in &["python", "python3"] {
        if let Ok(out) = Command::new(cmd).arg("--version").output() {
            if out.status.success() {
                let ver = String::from_utf8_lossy(&out.stdout);
                let ver = if ver.trim().is_empty() {
                    String::from_utf8_lossy(&out.stderr)
                } else {
                    ver
                };
                let trimmed = ver.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

fn find_blender() -> Option<PathBuf> {
    // 1. Check PATH
    if let Ok(out) = Command::new("blender").arg("--version").output() {
        if out.status.success() {
            return Some(PathBuf::from("blender"));
        }
    }

    // 2. Check macOS standard paths
    #[cfg(target_os = "macos")]
    {
        let mac_paths = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.3.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.2.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.1.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.0.app/Contents/MacOS/Blender",
        ];
        for p in &mac_paths {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
    }

    // 3. Check Windows standard paths
    #[cfg(target_os = "windows")]
    {
        if let Ok(prog) = std::env::var("ProgramFiles") {
            let base = Path::new(&prog).join("Blender Foundation");
            if base.exists() {
                if let Ok(entries) = std::fs::read_dir(base) {
                    for entry in entries.flatten() {
                        let candidate = entry.path().join("blender.exe");
                        if candidate.exists() {
                            return Some(candidate);
                        }
                    }
                }
            }
        }
    }

    // 4. Check Linux standard paths
    #[cfg(target_os = "linux")]
    {
        for p in &["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"] {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

fn find_comfyui() -> Option<PathBuf> {
    let candidates = [
        ".",
        "..",
        "ComfyUI",
        "../ComfyUI",
        "../../ComfyUI",
        "ComfyUI-Easy-Install/ComfyUI",
        "../ComfyUI-Easy-Install/ComfyUI",
    ];

    for c in &candidates {
        let p = Path::new(c);
        if p.join("main.py").exists() || p.join("custom_nodes").exists() {
            if let Ok(canon) = p.canonicalize() {
                return Some(canon);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_probe_runs_without_panic() {
        let env = EnvironmentInfo::probe();
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert!(env.recommended_wtivo_res >= 256 && env.recommended_wtivo_res <= 512);
        assert!(env.recommended_proxy_points > 0);

        let diag = env.format_diagnostic();
        assert!(diag.contains("Watertight 3D System & Environment Diagnostics"));
        assert!(diag.contains(&env.os));
    }

    #[test]
    fn test_detect_hardware_convenience() {
        let hw = detect_hardware();
        assert!(!hw.os.is_empty());
    }
}

