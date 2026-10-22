//! 3D Mesh representation and standard file format I/O (OBJ, STL, PLY).

use crate::error::{Result, WatertightError};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// 3D Axis-Aligned Bounding Box
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub fn new() -> Self {
        Self {
            min: [f64::INFINITY, f64::INFINITY, f64::INFINITY],
            max: [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        }
    }

    pub fn extend(&mut self, p: [f64; 3]) {
        for (i, &coord) in p.iter().enumerate() {
            if coord < self.min[i] {
                self.min[i] = coord;
            }
            if coord > self.max[i] {
                self.max[i] = coord;
            }
        }
    }

    pub fn size(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn max_extent(&self) -> f64 {
        let s = self.size();
        s[0].max(s[1]).max(s[2])
    }

    pub fn diagonal(&self) -> f64 {
        let s = self.size();
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::new()
    }
}

/// 3D Point alias
pub type Point3 = [f64; 3];
/// 3D Vector alias
pub type Vector3 = [f64; 3];
/// Triangular face index triplet
pub type Face = [usize; 3];

/// Triangular 3D Mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[usize; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub uvs: Vec<[f64; 2]>,
}

impl Mesh {
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Self {
        let mut mesh = Self {
            vertices,
            faces,
            normals: Vec::new(),
            uvs: Vec::new(),
        };
        mesh.compute_vertex_normals();
        mesh
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    pub fn bounding_box(&self) -> Aabb {
        let mut aabb = Aabb::new();
        for &v in &self.vertices {
            aabb.extend(v);
        }
        aabb
    }

    /// Centers the mesh at origin [0, 0, 0] and scales to fit within target_extent
    pub fn normalize_scale(&mut self, target_extent: f64) {
        let aabb = self.bounding_box();
        let center = aabb.center();
        let extent = aabb.max_extent();
        let scale = if extent > 1e-12 {
            target_extent / extent
        } else {
            1.0
        };

        for v in &mut self.vertices {
            v[0] = (v[0] - center[0]) * scale;
            v[1] = (v[1] - center[1]) * scale;
            v[2] = (v[2] - center[2]) * scale;
        }
    }

    /// Computes normal vector for each triangle face
    pub fn compute_face_normals(&self) -> Vec<[f64; 3]> {
        self.faces
            .iter()
            .map(|&[i0, i1, i2]| {
                let (Some(&p0), Some(&p1), Some(&p2)) = (
                    self.vertices.get(i0),
                    self.vertices.get(i1),
                    self.vertices.get(i2),
                ) else {
                    return [0.0, 0.0, 1.0];
                };
                let v0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                let v1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                let n = [
                    v0[1] * v1[2] - v0[2] * v1[1],
                    v0[2] * v1[0] - v0[0] * v1[2],
                    v0[0] * v1[1] - v0[1] * v1[0],
                ];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-12 {
                    [n[0] / len, n[1] / len, n[2] / len]
                } else {
                    [0.0, 0.0, 1.0]
                }
            })
            .collect()
    }

    /// Computes angle-weighted or area-weighted vertex normals
    pub fn compute_vertex_normals(&mut self) {
        let mut normals = vec![[0.0, 0.0, 0.0]; self.vertices.len()];
        let face_normals = self.compute_face_normals();

        for (idx, &[i0, i1, i2]) in self.faces.iter().enumerate() {
            let fn_norm = face_normals[idx];
            for &vi in &[i0, i1, i2] {
                if let Some(n) = normals.get_mut(vi) {
                    n[0] += fn_norm[0];
                    n[1] += fn_norm[1];
                    n[2] += fn_norm[2];
                }
            }
        }

        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                *n = [0.0, 0.0, 1.0];
            }
        }

        self.normals = normals;
    }

    /// Load mesh from file (.obj, .stl, .ply, .glb)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "glb" || ext == "gltf" {
            let data = std::fs::read(path)?;
            return Self::from_glb(&data);
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        match ext.as_str() {
            "obj" => Self::from_obj(reader),
            "stl" => Self::from_stl(reader),
            "ply" => Self::from_ply(reader),
            other => Err(WatertightError::UnsupportedFormat(format!(
                "Unsupported file extension: .{}",
                other
            ))),
        }
    }

    /// Save mesh to file (.obj, .stl, .ply, .glb)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        match ext.as_str() {
            "obj" => self.to_obj(&mut writer),
            "stl" => self.to_stl_binary(&mut writer),
            "ply" => self.to_ply(&mut writer),
            "glb" => {
                let bytes = self.to_glb()?;
                writer.write_all(&bytes)?;
                Ok(())
            }
            other => Err(WatertightError::UnsupportedFormat(format!(
                "Unsupported file extension: .{}",
                other
            ))),
        }
    }

    // -------------------------------------------------------------
    // OBJ I/O
    // -------------------------------------------------------------

    pub fn from_obj<R: BufRead>(mut reader: R) -> Result<Self> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut line = String::new();

        while reader.read_line(&mut line)? > 0 {
            let trimmed = line.trim();
            if trimmed.starts_with("v ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 4 {
                    let x: f64 = parts[1].parse().unwrap_or(0.0);
                    let y: f64 = parts[2].parse().unwrap_or(0.0);
                    let z: f64 = parts[3].parse().unwrap_or(0.0);
                    vertices.push([x, y, z]);
                }
            } else if trimmed.starts_with("f ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 4 {
                    let mut face_v = Vec::new();
                    let mut valid = true;
                    for part in &parts[1..] {
                        let v_str = part.split('/').next().unwrap_or("");
                        match v_str.parse::<isize>() {
                            Ok(v_idx) => {
                                let idx = if v_idx > 0 {
                                    (v_idx - 1) as usize
                                } else {
                                    (vertices.len() as isize + v_idx) as usize
                                };
                                if idx < vertices.len() {
                                    face_v.push(idx);
                                } else {
                                    valid = false;
                                    break;
                                }
                            }
                            Err(_) => {
                                valid = false;
                                break;
                            }
                        }
                    }
                    // Fan triangulation for polygons
                    if valid && face_v.len() >= 3 {
                        for k in 1..face_v.len() - 1 {
                            faces.push([face_v[0], face_v[k], face_v[k + 1]]);
                        }
                    }
                }
            }
            line.clear();
        }

        Ok(Self::new(vertices, faces))
    }

    pub fn to_obj<W: Write>(&self, writer: &mut W) -> Result<()> {
        writeln!(writer, "# Exported by watertight-meshes-rs")?;
        for v in &self.vertices {
            writeln!(writer, "v {:.6} {:.6} {:.6}", v[0], v[1], v[2])?;
        }
        for n in &self.normals {
            writeln!(writer, "vn {:.6} {:.6} {:.6}", n[0], n[1], n[2])?;
        }
        for f in &self.faces {
            // 1-based indexing for OBJ
            writeln!(
                writer,
                "f {}//{} {}//{} {}//{}",
                f[0] + 1,
                f[0] + 1,
                f[1] + 1,
                f[1] + 1,
                f[2] + 1,
                f[2] + 1
            )?;
        }
        Ok(())
    }

    // -------------------------------------------------------------
    // STL I/O
    // -------------------------------------------------------------

    pub fn from_stl<R: Read + BufRead>(mut reader: R) -> Result<Self> {
        // Read first 80 bytes header to detect ASCII vs Binary
        let mut header = [0u8; 80];
        reader.read_exact(&mut header)?;

        let header_str = String::from_utf8_lossy(&header);
        if header_str.trim_start().starts_with("solid") {
            // Check if it's truly ASCII by reading rest as string
            let mut rest = Vec::new();
            reader.read_to_end(&mut rest)?;
            let mut full = header.to_vec();
            full.extend(rest);
            if let Ok(ascii_str) = String::from_utf8(full.clone()) {
                if ascii_str.contains("facet normal") && ascii_str.contains("endsolid") {
                    return Self::from_stl_ascii(&ascii_str);
                }
            }
            // If failed to parse as valid ASCII, fall back to Binary
            let mut cursor = std::io::Cursor::new(full);
            let mut h = [0u8; 80];
            cursor.read_exact(&mut h)?;
            return Self::from_stl_binary(&mut cursor);
        }

        Self::from_stl_binary(&mut reader)
    }

    fn from_stl_ascii(content: &str) -> Result<Self> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut current_tri = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("vertex ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 4 {
                    let x: f64 = parts[1].parse().unwrap_or(0.0);
                    let y: f64 = parts[2].parse().unwrap_or(0.0);
                    let z: f64 = parts[3].parse().unwrap_or(0.0);
                    let idx = vertices.len();
                    vertices.push([x, y, z]);
                    current_tri.push(idx);
                    if current_tri.len() == 3 {
                        faces.push([current_tri[0], current_tri[1], current_tri[2]]);
                        current_tri.clear();
                    }
                }
            }
        }

        Ok(Self::new(vertices, faces))
    }

    fn from_stl_binary<R: Read>(reader: &mut R) -> Result<Self> {
        let tri_count = reader.read_u32::<LittleEndian>()? as usize;
        let mut vertices = Vec::with_capacity(tri_count * 3);
        let mut faces = Vec::with_capacity(tri_count);

        for _ in 0..tri_count {
            // Skip 3 normal floats
            let _nx = reader.read_f32::<LittleEndian>()?;
            let _ny = reader.read_f32::<LittleEndian>()?;
            let _nz = reader.read_f32::<LittleEndian>()?;

            let p0 = [
                reader.read_f32::<LittleEndian>()? as f64,
                reader.read_f32::<LittleEndian>()? as f64,
                reader.read_f32::<LittleEndian>()? as f64,
            ];
            let p1 = [
                reader.read_f32::<LittleEndian>()? as f64,
                reader.read_f32::<LittleEndian>()? as f64,
                reader.read_f32::<LittleEndian>()? as f64,
            ];
            let p2 = [
                reader.read_f32::<LittleEndian>()? as f64,
                reader.read_f32::<LittleEndian>()? as f64,
                reader.read_f32::<LittleEndian>()? as f64,
            ];

            // 2-byte attribute byte count
            let _attr = reader.read_u16::<LittleEndian>()?;

            let idx = vertices.len();
            vertices.push(p0);
            vertices.push(p1);
            vertices.push(p2);
            faces.push([idx, idx + 1, idx + 2]);
        }

        Ok(Self::new(vertices, faces))
    }

    pub fn to_stl_binary<W: Write>(&self, writer: &mut W) -> Result<()> {
        let mut header = [0u8; 80];
        let title = b"watertight-meshes-rs binary STL export";
        header[..title.len()].copy_from_slice(title);
        writer.write_all(&header)?;

        writer.write_u32::<LittleEndian>(self.faces.len() as u32)?;

        let face_normals = self.compute_face_normals();

        for (i, &[i0, i1, i2]) in self.faces.iter().enumerate() {
            let n = face_normals[i];
            writer.write_f32::<LittleEndian>(n[0] as f32)?;
            writer.write_f32::<LittleEndian>(n[1] as f32)?;
            writer.write_f32::<LittleEndian>(n[2] as f32)?;

            for &idx in &[i0, i1, i2] {
                let p = self.vertices[idx];
                writer.write_f32::<LittleEndian>(p[0] as f32)?;
                writer.write_f32::<LittleEndian>(p[1] as f32)?;
                writer.write_f32::<LittleEndian>(p[2] as f32)?;
            }

            // Attribute byte count = 0
            writer.write_u16::<LittleEndian>(0)?;
        }

        Ok(())
    }

    // -------------------------------------------------------------
    // PLY I/O
    // -------------------------------------------------------------

    pub fn from_ply<R: BufRead>(mut reader: R) -> Result<Self> {
        let mut line = String::new();
        let mut num_vertices = 0;
        let mut num_faces = 0;

        // Parse header
        while reader.read_line(&mut line)? > 0 {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("element vertex ") {
                num_vertices = rest.parse::<usize>().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("element face ") {
                num_faces = rest.parse::<usize>().unwrap_or(0);
            } else if trimmed == "end_header" {
                break;
            }
            line.clear();
        }

        let mut vertices = Vec::with_capacity(num_vertices);
        for _ in 0..num_vertices {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let x: f64 = parts[0].parse().unwrap_or(0.0);
                let y: f64 = parts[1].parse().unwrap_or(0.0);
                let z: f64 = parts[2].parse().unwrap_or(0.0);
                vertices.push([x, y, z]);
            }
        }

        let mut faces = Vec::with_capacity(num_faces);
        for _ in 0..num_faces {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let count: usize = parts[0].parse().unwrap_or(0);
                if count >= 3 && parts.len() > count {
                    let mut idxs = Vec::with_capacity(count);
                    let mut valid = true;
                    for part in &parts[1..=count] {
                        let vi: usize = part.parse().unwrap_or(usize::MAX);
                        if vi < vertices.len() {
                            idxs.push(vi);
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        for k in 1..idxs.len() - 1 {
                            faces.push([idxs[0], idxs[k], idxs[k + 1]]);
                        }
                    }
                }
            }
        }

        Ok(Self::new(vertices, faces))
    }

    pub fn to_ply<W: Write>(&self, writer: &mut W) -> Result<()> {
        writeln!(writer, "ply")?;
        writeln!(writer, "format ascii 1.0")?;
        writeln!(writer, "comment Exported by watertight-meshes-rs")?;
        writeln!(writer, "element vertex {}", self.vertices.len())?;
        writeln!(writer, "property float x")?;
        writeln!(writer, "property float y")?;
        writeln!(writer, "property float z")?;
        writeln!(writer, "element face {}", self.faces.len())?;
        writeln!(writer, "property list uchar int vertex_indices")?;
        writeln!(writer, "end_header")?;

        for v in &self.vertices {
            writeln!(writer, "{:.6} {:.6} {:.6}", v[0], v[1], v[2])?;
        }
        for f in &self.faces {
            writeln!(writer, "3 {} {} {}", f[0], f[1], f[2])?;
        }

        Ok(())
    }

    // -------------------------------------------------------------
    // GLB (binary glTF) I/O
    // -------------------------------------------------------------

    /// Load a GLB (binary glTF) mesh, merging all mesh primitives.
    pub fn from_glb(data: &[u8]) -> Result<Self> {
        let gltf = gltf::Gltf::from_slice(data)
            .map_err(|e| WatertightError::FormatError(format!("Invalid GLB: {e}")))?;
        let blob: &[u8] = gltf.blob.as_deref().unwrap_or(&[]);

        let mut vertices: Vec<[f64; 3]> = Vec::new();
        let mut faces: Vec<[usize; 3]> = Vec::new();

        for mesh in gltf.document.meshes() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err(WatertightError::FormatError(format!(
                        "Unsupported primitive mode: {:?}",
                        primitive.mode()
                    )));
                }

                let reader = primitive.reader(|_buffer| Some(blob));
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or_else(|| {
                        WatertightError::FormatError("GLB primitive missing POSITION".into())
                    })?
                    .collect();
                let base = vertices.len();
                vertices.extend(positions.iter().map(|p| {
                    [p[0] as f64, p[1] as f64, p[2] as f64]
                }));

                if let Some(indices) = reader.read_indices() {
                    let idx: Vec<u32> = match indices {
                        gltf::mesh::util::ReadIndices::U8(it) => it.map(|i| i as u32).collect(),
                        gltf::mesh::util::ReadIndices::U16(it) => it.map(|i| i as u32).collect(),
                        gltf::mesh::util::ReadIndices::U32(it) => it.collect(),
                    };
                    for tri in idx.chunks(3) {
                        if tri.len() == 3 {
                            faces.push([
                                base + tri[0] as usize,
                                base + tri[1] as usize,
                                base + tri[2] as usize,
                            ]);
                        }
                    }
                } else {
                    // Non-indexed primitive: positions are sequential triangles.
                    for i in (0..positions.len()).step_by(3) {
                        if i + 2 < positions.len() {
                            faces.push([base + i, base + i + 1, base + i + 2]);
                        }
                    }
                }
            }
        }

        Ok(Mesh::new(vertices, faces))
    }

    /// Serialize the mesh to a GLB (binary glTF) byte buffer.
    pub fn to_glb(&self) -> Result<Vec<u8>> {
        // Pack positions (f32 x3) then indices (u32 x3) into the binary chunk.
        let mut bin = Vec::with_capacity(self.vertices.len() * 12 + self.faces.len() * 12);
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for v in &self.vertices {
            bin.extend_from_slice(&(v[0] as f32).to_le_bytes());
            bin.extend_from_slice(&(v[1] as f32).to_le_bytes());
            bin.extend_from_slice(&(v[2] as f32).to_le_bytes());
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        let pos_bytes = self.vertices.len() * 12;
        for f in &self.faces {
            for &idx in f {
                bin.extend_from_slice(&(idx as u32).to_le_bytes());
            }
        }
        let idx_bytes = self.faces.len() * 12;

        let mut position_accessor = serde_json::json!({
            "bufferView": 0,
            "componentType": 5126,
            "count": self.vertices.len(),
            "type": "VEC3"
        });
        if !self.vertices.is_empty() {
            position_accessor["min"] = serde_json::json!([min[0], min[1], min[2]]);
            position_accessor["max"] = serde_json::json!([max[0], max[1], max[2]]);
        }

        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
            "buffers": [{"byteLength": bin.len()}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": pos_bytes, "target": 34962},
                {"buffer": 0, "byteOffset": pos_bytes, "byteLength": idx_bytes, "target": 34963}
            ],
            "accessors": [
                position_accessor,
                {"bufferView": 1, "componentType": 5125, "count": self.faces.len() * 3, "type": "SCALAR"}
            ]
        });

        let json_bytes = serde_json::to_vec(&json)?;

        // Assemble the GLB container: header + JSON chunk + BIN chunk.
        let json_padded = (json_bytes.len() + 3) & !3;
        let bin_padded = (bin.len() + 3) & !3;
        let total = 12 + 8 + json_padded + 8 + bin_padded;

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&0x4654_6C67u32.to_le_bytes()); // "glTF"
        out.extend_from_slice(&2u32.to_le_bytes()); // version 2
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_padded as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // "JSON"
        out.extend_from_slice(&json_bytes);
        while out.len() % 4 != 0 {
            out.push(0x20);
        }
        out.extend_from_slice(&(bin_padded as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // "BIN\0"
        out.extend_from_slice(&bin);
        while out.len() % 4 != 0 {
            out.push(0x00);
        }

        Ok(out)
    }

    // -------------------------------------------------------------
    // Primitives for testing & verification
    // -------------------------------------------------------------

    /// Generates a watertight cube centered at origin with side length `size`.
    /// 8 vertices, 12 triangular faces.
    pub fn cube(size: f64) -> Self {
        let s = size * 0.5;
        let vertices = vec![
            [-s, -s, -s], // 0
            [s, -s, -s],  // 1
            [s, s, -s],   // 2
            [-s, s, -s],  // 3
            [-s, -s, s],  // 4
            [s, -s, s],   // 5
            [s, s, s],    // 6
            [-s, s, s],   // 7
        ];

        let faces = vec![
            // Front (-Z)
            [0, 2, 1],
            [0, 3, 2],
            // Back (+Z)
            [4, 5, 6],
            [4, 6, 7],
            // Left (-X)
            [0, 4, 7],
            [0, 7, 3],
            // Right (+X)
            [1, 2, 6],
            [1, 6, 5],
            // Top (+Y)
            [3, 7, 6],
            [3, 6, 2],
            // Bottom (-Y)
            [0, 1, 5],
            [0, 5, 4],
        ];

        Self::new(vertices, faces)
    }

    /// Generates a cylinder. If `capped == false`, creates an open cylinder with 2 boundary loops.
    pub fn cylinder(radius: f64, height: f64, segments: usize, capped: bool) -> Self {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let h2 = height * 0.5;

        // Bottom ring (0..segments)
        for i in 0..segments {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
            vertices.push([radius * theta.cos(), radius * theta.sin(), -h2]);
        }
        // Top ring (segments..2*segments)
        for i in 0..segments {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (segments as f64);
            vertices.push([radius * theta.cos(), radius * theta.sin(), h2]);
        }

        // Side quads as pairs of triangles
        for i in 0..segments {
            let next = (i + 1) % segments;
            let b0 = i;
            let b1 = next;
            let t0 = i + segments;
            let t1 = next + segments;

            faces.push([b0, b1, t0]);
            faces.push([b1, t1, t0]);
        }

        if capped {
            let b_center = vertices.len();
            vertices.push([0.0, 0.0, -h2]);
            let t_center = vertices.len();
            vertices.push([0.0, 0.0, h2]);

            for i in 0..segments {
                let next = (i + 1) % segments;
                // Bottom cap
                faces.push([b_center, (i + 1) % segments, i]);
                // Top cap
                faces.push([t_center, i + segments, next + segments]);
            }
        }

        Self::new(vertices, faces)
    }

    /// Generates a known non-manifold T-junction (3 faces sharing an edge).
    pub fn non_manifold_t_junction() -> Self {
        let vertices = vec![
            [0.0, 0.0, 0.0], // 0: bottom of shared edge
            [0.0, 1.0, 0.0], // 1: top of shared edge
            [1.0, 0.5, 0.0], // 2: wing 1
            [-1.0, 0.5, 0.0], // 3: wing 2
            [0.0, 0.5, 1.0], // 4: wing 3
        ];
        let faces = vec![
            [0, 1, 2], // Face 1 shares (0, 1)
            [0, 1, 3], // Face 2 shares (0, 1)
            [0, 1, 4], // Face 3 shares (0, 1) -> 3 faces sharing 1 edge!
        ];
        Self::new(vertices, faces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_cube_primitive() {
        let cube = Mesh::cube(2.0);
        assert_eq!(cube.vertex_count(), 8);
        assert_eq!(cube.face_count(), 12);
        let aabb = cube.bounding_box();
        assert!((aabb.size()[0] - 2.0).abs() < 1e-9);
        assert!((aabb.size()[1] - 2.0).abs() < 1e-9);
        assert!((aabb.size()[2] - 2.0).abs() < 1e-9);
        assert!((aabb.center()[0]).abs() < 1e-9);
    }

    #[test]
    fn test_cylinder_primitive() {
        let open_cyl = Mesh::cylinder(1.0, 4.0, 16, false);
        assert_eq!(open_cyl.vertex_count(), 32);
        assert_eq!(open_cyl.face_count(), 32);

        let capped_cyl = Mesh::cylinder(1.0, 4.0, 16, true);
        assert_eq!(capped_cyl.vertex_count(), 34);
        assert_eq!(capped_cyl.face_count(), 64);
    }

    #[test]
    fn test_normalize_scale() {
        let mut cube = Mesh::cube(10.0);
        cube.normalize_scale(2.0);
        let aabb = cube.bounding_box();
        assert!((aabb.max_extent() - 2.0).abs() < 1e-9);
        assert!(aabb.center()[0].abs() < 1e-9);
    }

    #[test]
    fn test_compute_normals() {
        let cube = Mesh::cube(1.0);
        let face_normals = cube.compute_face_normals();
        assert_eq!(face_normals.len(), 12);
        for n in &face_normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_obj_roundtrip() {
        let original = Mesh::cube(1.0);
        let temp_path = std::env::temp_dir().join(format!("test_cube_{}.obj", std::process::id()));

        original.save(&temp_path).expect("save OBJ failed");
        let loaded = Mesh::load(&temp_path).expect("load OBJ failed");

        assert_eq!(loaded.vertex_count(), original.vertex_count());
        assert_eq!(loaded.face_count(), original.face_count());

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_stl_roundtrip() {
        let original = Mesh::cube(1.0);
        let temp_path = std::env::temp_dir().join(format!("test_cube_{}.stl", std::process::id()));

        original.save(&temp_path).expect("save STL failed");
        let loaded = Mesh::load(&temp_path).expect("load STL failed");

        assert_eq!(loaded.face_count(), original.face_count());
        // In STL, vertices are per-triangle (3 * 12 = 36)
        assert_eq!(loaded.vertex_count(), 36);

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_ply_roundtrip() {
        let original = Mesh::cube(1.0);
        let temp_path = std::env::temp_dir().join(format!("test_cube_{}.ply", std::process::id()));

        original.save(&temp_path).expect("save PLY failed");
        let loaded = Mesh::load(&temp_path).expect("load PLY failed");

        assert_eq!(loaded.vertex_count(), original.vertex_count());
        assert_eq!(loaded.face_count(), original.face_count());

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_obj_out_of_range_face_index_does_not_panic() {
        // Face references vertex 999 which does not exist
        let data = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 999\n";
        let reader = std::io::BufReader::new(std::io::Cursor::new(&data[..]));
        let mesh = Mesh::from_obj(reader).expect("must not panic on out-of-range index");
        assert_eq!(mesh.face_count(), 0, "invalid face must be skipped");
    }

    #[test]
    fn test_obj_negative_face_index_does_not_panic() {
        let data = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -999\n";
        let reader = std::io::BufReader::new(std::io::Cursor::new(&data[..]));
        let mesh = Mesh::from_obj(reader).expect("must not panic on out-of-range negative index");
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn test_ply_out_of_range_face_index_does_not_panic() {
        let data = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 99\n";
        let reader = std::io::BufReader::new(std::io::Cursor::new(&data[..]));
        let mesh = Mesh::from_ply(reader).expect("must not panic on out-of-range index");
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn test_ply_quad_face_is_triangulated() {
        // A quad face (4 vertices) must fan-triangulate into 2 triangles.
        let data = b"ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n";
        let reader = std::io::BufReader::new(std::io::Cursor::new(&data[..]));
        let mesh = Mesh::from_ply(reader).expect("parse quad");
        assert_eq!(mesh.face_count(), 2, "quad must fan-triangulate into 2 triangles");
    }

    #[test]
    fn test_glb_roundtrip() {
        let original = Mesh::cube(2.0);
        let temp_path = std::env::temp_dir().join(format!("test_cube_{}.glb", std::process::id()));

        original.save(&temp_path).expect("save GLB failed");
        let loaded = Mesh::load(&temp_path).expect("load GLB failed");

        assert_eq!(loaded.vertex_count(), original.vertex_count());
        assert_eq!(loaded.face_count(), original.face_count());

        let _ = fs::remove_file(temp_path);
    }
}


