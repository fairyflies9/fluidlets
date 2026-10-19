"""Hand-written VTK XML output (ASCII), readable by ParaView and VisIt.

- ``write_vtp``: Lagrangian points (PolyData) with point data, e.g. force.
- ``write_vtr``: fields on a rectilinear evaluation grid (RectilinearGrid).
- ``write_pvd``: a collection file tying a time series together.

Point data are given as a dict name -> array of shape (N,) or (N, 3).
2D data are padded with z = 0 so ParaView treats them uniformly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_HEADER = '<?xml version="1.0"?>\n<VTKFile type="{t}" version="1.0" byte_order="LittleEndian">\n'


def _numbers(a) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(a))


def _pad3(a: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(a, float))
    return a if a.shape[1] == 3 else np.c_[a, np.zeros((a.shape[0], 3 - a.shape[1]))]


def _point_data(fields: dict | None, n: int) -> str:
    if not fields:
        return ""
    out = ["<PointData>"]
    for name, values in fields.items():
        values = np.asarray(values)
        if np.iscomplexobj(values):
            raise ValueError(f"{name}: write real and imaginary parts as separate fields")
        if values.ndim == 1:
            ncomp = 1
        else:
            values = _pad3(values)
            ncomp = 3
        if values.shape[0] != n:
            raise ValueError(f"{name}: expected {n} values, got {values.shape[0]}")
        out.append(f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">')
        out.append(_numbers(values))
        out.append("</DataArray>")
    out.append("</PointData>")
    return "\n".join(out) + "\n"


def write_vtp(path, points, fields: dict | None = None) -> None:
    """Points as vertices, with optional point data."""
    pts = _pad3(points)
    n = pts.shape[0]
    with open(path, "w") as fh:
        fh.write(_HEADER.format(t="PolyData"))
        fh.write(f'<PolyData>\n<Piece NumberOfPoints="{n}" NumberOfVerts="{n}">\n')
        fh.write(_point_data(fields, n))
        fh.write('<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        fh.write(_numbers(pts) + "\n</DataArray>\n</Points>\n")
        fh.write('<Verts>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        fh.write(" ".join(map(str, range(n))) + "\n</DataArray>\n")
        fh.write('<DataArray type="Int64" Name="offsets" format="ascii">\n')
        fh.write(" ".join(map(str, range(1, n + 1))) + "\n</DataArray>\n</Verts>\n")
        fh.write("</Piece>\n</PolyData>\n</VTKFile>\n")


def write_vtr(path, x, y, z, fields: dict) -> None:
    """Fields on the grid x (fastest index) by y by z.

    Each field has shape (len(x)*len(y)*len(z),) or (..., 3), ordered with x
    varying fastest, then y, then z, i.e. what
    ``np.meshgrid(z, y, x, indexing="ij")`` then ``.ravel()`` produces.
    """
    x, y, z = (np.atleast_1d(np.asarray(c, float)) for c in (x, y, z))
    nx, ny, nz = len(x), len(y), len(z)
    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    with open(path, "w") as fh:
        fh.write(_HEADER.format(t="RectilinearGrid"))
        fh.write(f'<RectilinearGrid WholeExtent="{extent}">\n<Piece Extent="{extent}">\n')
        fh.write(_point_data(fields, nx * ny * nz))
        fh.write("<Coordinates>\n")
        for c in (x, y, z):
            fh.write(f'<DataArray type="Float64" format="ascii">\n{_numbers(c)}\n</DataArray>\n')
        fh.write("</Coordinates>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n")


def write_pvd(path, entries) -> None:
    """Collection file; ``entries`` is a list of (time, relative file name)."""
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(_HEADER.format(t="Collection"))
        fh.write("<Collection>\n")
        for t, name in entries:
            fh.write(f'<DataSet timestep="{t!r}" part="0" file="{name}"/>\n')
        fh.write("</Collection>\n</VTKFile>\n")
