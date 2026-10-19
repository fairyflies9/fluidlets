"""Input parser, .vertex files and VTK output."""
import os
import shutil
import subprocess
import xml.dom.minidom
from pathlib import Path

import numpy as np
import pytest

from fluidlets.io import read_input, read_vertex, write_pvd, write_vertex, write_vtp, write_vtr

IB2D_STYLE = """%*****% INPUT2d FILE %*****%
Fluid_Parameters {
mu = 6.667            % Dynamic Viscosity (N*s/m^2)
rho = 1000            % Density (kg/m^3)
}
Lag_Name {
string_name = "jelly1024"  % String name for simulation
}
"""

FLAT_STYLE = """# fluidlets flat style
kernel  = stokeslet   # comment
eps     = 0.5
grid_x  = -1 8 32
"""


def test_read_ib2d_and_flat_styles(tmp_path):
    (tmp_path / "input2d").write_text(IB2D_STYLE)
    p = read_input(tmp_path / "input2d")
    assert p == {"mu": 6.667, "rho": 1000, "string_name": "jelly1024"}
    assert isinstance(p["rho"], int)
    (tmp_path / "input3d").write_text(FLAT_STYLE)
    q = read_input(tmp_path / "input3d")
    assert q["kernel"] == "stokeslet" and q["eps"] == 0.5
    assert np.array_equal(q["grid_x"], [-1, 8, 32])


def test_vertex_round_trip(tmp_path):
    pts = np.random.default_rng(0).normal(size=(17, 3))
    write_vertex(tmp_path / "a.vertex", pts)
    assert np.array_equal(read_vertex(tmp_path / "a.vertex"), pts)
    (tmp_path / "b.vertex").write_text("2\n0.5 1.0\n2.0 3.0\n")         # IB2d 2D file
    assert read_vertex(tmp_path / "b.vertex").shape == (2, 2)


def write_samples(folder):
    rng = np.random.default_rng(1)
    pts, f = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    x, y, z = np.linspace(0, 1, 4), np.linspace(-1, 1, 3), np.linspace(0, 2, 2)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    u = np.c_[X.ravel(), 2 * Y.ravel(), 3 * Z.ravel()]
    write_vtp(folder / "lag.vtp", pts, {"force": f, "id": np.arange(5.0)})
    write_vtr(folder / "eul.vtr", x, y, z, {"u": u})
    write_pvd(folder / "run.pvd", [(0.0, "lag.vtp")])
    return pts, f, u


def test_vtk_files_are_well_formed_xml(tmp_path):
    write_samples(tmp_path)
    for name in ["lag.vtp", "eul.vtr", "run.pvd"]:
        xml.dom.minidom.parse(str(tmp_path / name))


PVPYTHON = shutil.which("pvpython") or next(
    (str(p) for p in Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob("ParaView*/bin/pvpython.exe")),
    None,
)

READ_BACK = """
import sys, numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
folder = sys.argv[1]
def read(reader_cls, name):
    r = reader_cls(); r.SetFileName(folder + '/' + name); r.Update()
    return r.GetOutput()
# plain VTK readers (what ParaView uses to open the files); paraview.simple's
# servermanager.Fetch returns a mangled rectilinear grid in ParaView 5.10
p = read(vtk.vtkXMLPolyDataReader, 'lag.vtp')
g = read(vtk.vtkXMLRectilinearGridReader, 'eul.vtr')
copy = lambda a: np.array(vtk_to_numpy(a), copy=True)
np.savez(folder + '/back.npz', pts=copy(p.GetPoints().GetData()), force=copy(p.GetPointData().GetArray('force')),
         u=copy(g.GetPointData().GetArray('u')), dims=np.array(g.GetDimensions()),
         gpts=np.array([g.GetPoint(i) for i in range(g.GetNumberOfPoints())]))
"""


@pytest.mark.skipif(PVPYTHON is None, reason="ParaView (pvpython) not installed")
def test_paraview_reads_back_our_vtk(tmp_path):
    pts, f, u = write_samples(tmp_path)
    (tmp_path / "read_back.py").write_text(READ_BACK)
    subprocess.run([PVPYTHON, str(tmp_path / "read_back.py"), str(tmp_path)], check=True, capture_output=True)
    back = np.load(tmp_path / "back.npz")
    assert np.allclose(back["pts"], pts) and np.allclose(back["force"], f)
    assert list(back["dims"]) == [4, 3, 2]
    assert np.allclose(back["u"], u)
    # u was built as (x, 2y, 3z) from the grid coordinates, so ordering is checked too
    gp = back["gpts"]
    assert np.allclose(back["u"], gp * [1, 2, 3])
