from __future__ import annotations
import mrcfile
import numpy as np

def read_map(path: str):
    """Return (data, voxel_size, origin)."""
    with mrcfile.open(path, permissive=True) as m:
        return np.asarray(m.data, dtype=np.float32), m.voxel_size, m.header.origin

def write_map(path: str, data: np.ndarray, voxel_size, origin) -> None:
    """Write an .mrc, preserving voxel size and origin so it stays aligned in viewers."""
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data.astype(np.float32))
        m.voxel_size = voxel_size
        m.header.origin = origin