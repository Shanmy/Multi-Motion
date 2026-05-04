# SMPL-A Model

SMPL-A Model from BEV which adds infant and child meshes into the SMPL model. Try to use this SMPL model for all applications.

## Format

* `betas`: 11-dim shape parameter giving the mesh shape. Setting all values to 0 gives the neutral mesh.
* `thetas`: 72-dim pose parameter. 24 joints × 3-dim angle vector for each joint.
* `trans`: 3-d global translation of mesh root joint. Gives location in 3D space.

## Data Files

The `.pth` and `.npy` files in this directory are **placeholder stubs** and must be replaced with real model weights before use:

| File | Description |
|------|-------------|
| `smpla_packed_info.pth` | SMPL-A body model (adults + infants/children) |
| `smil_packed_info.pth` | SMIL infant body model |
| `smpl_faces.npy` | Mesh face indices for SMPL vertices |

### How to obtain

These files come from the [ROMP / BEV](https://github.com/Arthur151/ROMP) project:

1. Install `simple_romp`: `pip install simple_romp`
2. Run `bev --video demo.mp4` once to trigger the automatic model download, then locate the cached `.pth` files (typically at `~/.romp/` or printed during the first run).
3. Copy `smpla_packed_info.pth`, `smil_packed_info.pth`, and `smpl_faces.npy` into this directory.

Alternatively, download directly from the [ROMP releases page](https://github.com/Arthur151/ROMP/releases) and look for model asset archives.
