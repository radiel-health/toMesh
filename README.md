# toMesh — Cardiovascular Mesh Editor

toMesh is an end-to-end pipeline that takes a raw CT scan (DICOM or NIfTI), automatically
segments cardiovascular structures using TotalSegmentator, generates a clean triangular
surface mesh, lets a clinician or engineer interactively edit that mesh and tag boundary
conditions (inlet / outlet / wall) in a PyQt6 GUI, and finally exports a PyTorch Geometric
graph ready to drop into a CFD machine-learning surrogate model.

---

## Installation

### pip

```bash
pip install -r requirements.txt
# PyTorch Geometric also requires matching torch/CUDA wheels; see https://pyg.org/install
```

### conda

```bash
conda env create -f environment.yml
conda activate tomesh
```

---

## Quickstart

### Full end-to-end (CT → graph)

```bash
python pipeline.py \
  --input  /path/to/ct.nii.gz \
  --output /path/to/output/ \
  --structures aorta,pulmonary_artery \
  --launch-gui
```

### Skip segmentation, use existing mask

```bash
python pipeline.py \
  --input /path/to/ct.nii.gz \
  --output /path/to/output/ \
  --skip-segment \
  --mask-path /path/to/combined_mask.nii.gz \
  --launch-gui
```

### Headless batch export (no GUI)

```bash
python pipeline.py \
  --input /path/to/ct.nii.gz \
  --output /path/to/output/ \
  --skip-gui
```

### Re-export graph from an already-edited .vtp

```bash
python pipeline.py \
  --input /path/to/ct.nii.gz \
  --output /path/to/output/ \
  --export-only \
  --mask-path /path/to/mesh_with_bcs.vtp
```

### Launch GUI standalone

```bash
python -m gui /path/to/mesh.vtp
```

---

## Graph Output Format

The exported `graph.pt` is a `torch_geometric.data.Data` object compatible with PyG GNN
surrogates. The Bifurcation WSS surrogate (sibling repo) expects exactly this layout.

### Node features — `data.x`  shape `(N, 9)`

| Index | Feature       | Description                                  |
|-------|---------------|----------------------------------------------|
| 0–2   | x, y, z       | Position, normalized to unit bounding box    |
| 3–5   | nx, ny, nz    | Surface normal unit vector                   |
| 6     | mean_curv     | Mean curvature (scalar)                      |
| 7     | is_inlet      | BC one-hot: 1 if inlet, else 0               |
| 8     | is_outlet     | BC one-hot: 1 if outlet, else 0              |
| (implicit) | is_wall  | Derived: not inlet and not outlet            |

### Edge connectivity — `data.edge_index`  shape `(2, E)`

Two edge sources merged and deduplicated:
1. **Face-adjacency**: vertices sharing a triangle connected
2. **Radius neighbors**: vertices within `3 × mean_edge_length` connected

### Edge features — `data.edge_attr`  shape `(E, 5)`

| Index | Feature    | Description                          |
|-------|------------|--------------------------------------|
| 0–2   | dx, dy, dz | Relative position vector (src → dst) |
| 3     | dist       | Euclidean distance                   |
| 4     | normal_ang | Angle between endpoint normals (rad) |

### Graph-level metadata (extra `Data` attributes)

| Attribute          | Type    | Description                              |
|--------------------|---------|------------------------------------------|
| original_scale     | Tensor  | Bounding-box dims before normalization   |
| original_centroid  | Tensor  | Centroid before normalization            |
| n_inlet_faces      | int     | Number of inlet-tagged faces             |
| n_outlet_faces     | int     | Number of outlet-tagged faces            |
| n_wall_faces       | int     | Number of wall-tagged faces              |
| source_file        | str     | Path to source CT scan                   |
| mesh_face_count    | int     | Triangle count in final mesh             |
| mesh_vertex_count  | int     | Vertex count in final mesh               |

---

## GUI Screenshot

[screenshot]

---

## Known Limitations

- **TotalSegmentator GPU memory**: The `total` task requires ~8 GB VRAM. Use `--fast` in
  TotalSegmentator config for GPUs with less memory (reduces accuracy slightly).
- **Sculpt tool**: The brush-based sculpt is functional but not production-grade. Large
  brushes on dense meshes may be slow due to Python-level vertex iteration.
- **Self-intersection detection**: Uses PyVista's collision detection which can be slow on
  meshes > 50k faces; the check is run as a warning, not a hard block.
- **DICOM support**: DICOM to NIfTI conversion uses SimpleITK. Multi-frame DICOMs from some
  scanner manufacturers may require additional series-selection logic.
- **BC auto-detect heuristic**: The Z-position heuristic (lowest free-edge loop = inlet)
  works well for aorta/vessel models but may need manual override for complex geometries.
- **Windows paths**: The GUI and CLI have been tested on Linux/macOS. Windows path handling
  should work but is less tested.
