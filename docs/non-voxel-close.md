# Non-voxel close

Feature. A repair that seals a mesh by editing its own triangles, and stops when that is not enough.

The voxel remesh in `voxel.rs` is a new surface. `auto_repair` calls it whenever weld and `fill_holes` leave the mesh open or non-manifold (`src/lib.rs`). That is why the result looks voxel. `watertight repair --no-remesh` already skips that call, then still exits 0 and writes the mesh.

## Outcome

`close` runs weld, degenerate-face removal, duplicate-face removal, non-manifold edge reduction, boundary fill, and a global normal flip when the closed mesh has negative signed volume. It never calls `voxelize_and_remesh`.

Success is `analyze_topology` reporting `is_watertight`: no boundary edges, no non-manifold edges or vertices, signed volume > 1e-9. That is the definition already in `src/topology.rs`.

Failure returns the patched mesh plus the health report and a non-zero exit. It does not fall through to Marching Cubes.

## Surface rule

- A vertex that came from the input stays at its welded position. Weld tolerance is the only allowed move, and it only merges vertices already within that distance.
- New vertices are allowed only where `fill_holes` already adds a centroid for a boundary loop it cannot ear-clip.
- Faces may be added only on those boundary loops. Faces may be removed when they are degenerate, exact duplicates (same three vertex indices), or the fin chosen by the edge rule below.
- Decimation is off on this path. QEM moves vertices.

## In scope

1. Spatial-hash weld and degenerate purge (existing `merge_by_distance`, `remove_degenerate_faces`).
2. Boundary loops whose edges have exactly one incident face (existing `fill_holes`). A loop that is already a single triangle is deleted instead of filled, because filling it stacks a second copy on that triangle.
3. Exact duplicate faces.
4. Edges with 3 or more faces, including `Mesh::non_manifold_t_junction`. The auditor marks an edge non-manifold only when `faces.len() > 2` (`src/topology.rs`). Reduce each such edge to 2 faces by deleting fins. The edge rule does not move vertices.
5. Non-manifold vertices. Each extra fan is given a copy of that vertex at the same position, after welding, so the copy is not merged again.
6. One winding flip of a mesh that is already closed and manifold but has negative signed volume (`has_inverted_normals`).

### Edge rule

Process every edge with 3 or more incident faces. Delete one incident face at a time until two remain, then run `fill_holes`.

Delete, in order:

1. An exact duplicate of another face on that edge.
2. Otherwise the face with the most other edges that currently have exactly one incident face. A fin stuck to a closed body scores high. A face that is part of that body scores zero on those edges.
3. Tie-break: smaller area, then greater face index.

This clears the edge counter. It does not invent a volume. A flat 3-wing sheet stays flat after one wing is dropped, so `signed_volume` stays ~0 and `is_watertight` stays false. A cube with a third face on one edge loses the fin and stays the cube.

## Out of scope

These stay unresolved on purpose. Fixing them builds a new surface or deletes real parts.

- Cracks wider than the weld tolerance. They are not one boundary loop of coincident vertices.
- Geometric self-intersections. The auditor does not detect them. A topologically closed self-intersecting mesh can still report `is_watertight`.
- Separate closed shells (interior cavities, floating parts, accessories). Voxel flood-fill deletes or fills them. This path keeps them.
- A guarantee of slicer success beyond the topological checks above.
- Replacing `remesh`. That command stays the voxel path.

## Acceptance

Each item is a `cargo test` assertion or a CLI run on a fixture. Positions are compared to the input after weld.

| Case | Pass condition |
|---|---|
| Open cylinder (`Mesh::cylinder(..., false)`) | `is_watertight`. Side-wall vertex positions unchanged. Only cap vertices or cap faces are new. |
| Closed cube | Face count unchanged. Positions identical. |
| Inverted closed cube | Positions identical. `signed_volume > 0` after one flip. |
| T-junction fixture (`Mesh::non_manifold_t_junction`) | `non_manifold_edge_count == 0`. Kept input positions unchanged. No voxel grid. `is_watertight` stays false (no thickness). |
| Cube plus one fin on an existing edge | Fin face removed. `is_watertight`. Cube vertex positions identical. |
| Two cubes joined at one welded corner | `non_manifold_vertex_count == 0` and `is_watertight`. 24 faces. The shared corner is duplicated in place. |
| `repair --no-remesh` on an open mesh | Does not print the voxel-remesh log line. Exit status matches `is_watertight`. |

## Implementation

`close_surface` in `src/close.rs` is the close. `watertight repair --no-remesh` calls it and does not decimate. `remesh` is unchanged. `--no-fill-holes` still skips boundary fill.
