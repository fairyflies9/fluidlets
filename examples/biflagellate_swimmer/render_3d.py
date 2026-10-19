"""Render the frames saved by movie_3d.py. Runs under ParaView's pvpython (uses its VTK).

Usage: pvpython render_3d.py FOLDER [FRAME]   (FRAME renders a single test image)
"""
import sys, glob, math
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

folder = sys.argv[1]
only = int(sys.argv[2]) if len(sys.argv) > 2 else None
files = sorted(glob.glob(folder + "/frame*.npz"))
if only is not None:
    files = [files[only]]

all_files = sorted(glob.glob(folder + "/frame*.npz"))
TRAIL = 12            # frames of trail behind each tracer
VMAX = 0.2

lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(256)
lut.SetRange(0, VMAX)
# viridis anchor colours, interpolated
ctf = vtk.vtkColorTransferFunction()
for x, rgb in [(0.0, (0.267, 0.005, 0.329)), (0.25, (0.230, 0.322, 0.546)), (0.5, (0.128, 0.567, 0.551)),
               (0.75, (0.369, 0.789, 0.383)), (1.0, (0.993, 0.906, 0.144))]:
    ctf.AddRGBPoint(x, *rgb)
for i in range(256):
    lut.SetTableValue(i, *ctf.GetColor(i / 255), 1.0)
lut.Build()


def polyline(P):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(P), deep=True))
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(P))
    for i in range(len(P)):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray(); cells.InsertNextCell(line)
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(cells)
    return pd


def tube_actor(pd, radius, color):
    t = vtk.vtkTubeFilter(); t.SetInputData(pd); t.SetRadius(radius); t.SetNumberOfSides(16); t.CappingOn()
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(t.GetOutputPort()); m.ScalarVisibilityOff()
    a = vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(*color)
    a.GetProperty().SetSpecular(0.3)
    return a


ren = vtk.vtkRenderer()
win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(900, 760)
win.SetMultiSamples(8)

for n, f in enumerate(files):
    d = np.load(f)
    k = int(f[-7:-4])
    ren.RemoveAllViewProps()
    ren.GradientBackgroundOn(); ren.SetBackground(0.10, 0.11, 0.14); ren.SetBackground2(0.30, 0.33, 0.40)
    x0, B, ax = d["x0"], d["B"], d["axes"]

    # flow field
    gx, gy, gz, u = d["gx"], d["gy"], d["gz"], d["u"]
    grid = vtk.vtkRectilinearGrid(); grid.SetDimensions(len(gx), len(gy), len(gz))
    grid.SetXCoordinates(numpy_to_vtk(gx, deep=True)); grid.SetYCoordinates(numpy_to_vtk(gy, deep=True))
    grid.SetZCoordinates(numpy_to_vtk(gz, deep=True))
    vel = numpy_to_vtk(np.ascontiguousarray(u), deep=True); vel.SetName("u")
    grid.GetPointData().SetVectors(vel)
    spd = numpy_to_vtk(np.linalg.norm(u, axis=1), deep=True); spd.SetName("speed")
    grid.GetPointData().AddArray(spd)

    # tracer particles, with trails through their last few positions
    tr = d["tracers"]
    pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(tr), deep=True))
    cloud = vtk.vtkPolyData(); cloud.SetPoints(pts)
    sp = numpy_to_vtk(np.ascontiguousarray(d["tracer_speed"]), deep=True); sp.SetName("speed")
    cloud.GetPointData().SetScalars(sp)
    ball = vtk.vtkSphereSource(); ball.SetRadius(0.018); ball.SetThetaResolution(10); ball.SetPhiResolution(10)
    glyph = vtk.vtkGlyph3D(); glyph.SetInputData(cloud); glyph.SetSourceConnection(ball.GetOutputPort())
    glyph.SetScaleModeToDataScalingOff(); glyph.SetColorModeToColorByScalar()
    gm = vtk.vtkPolyDataMapper(); gm.SetInputConnection(glyph.GetOutputPort())
    gm.SetLookupTable(lut); gm.SetScalarRange(0, VMAX); gm.UseLookupTableScalarRangeOn()
    ga = vtk.vtkActor(); ga.SetMapper(gm); ga.GetProperty().SetSpecular(0.3)
    ren.AddActor(ga)
    history = [np.load(g)["tracers"] for g in all_files[max(0, k - TRAIL): k + 1]]
    if len(history) > 1:
        H = np.stack(history, axis=1)                 # (tracers, trail length, 3)
        tp = vtk.vtkPoints(); tp.SetData(numpy_to_vtk(np.ascontiguousarray(H.reshape(-1, 3)), deep=True))
        lines = vtk.vtkCellArray()
        L = H.shape[1]
        for i in range(H.shape[0]):
            line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(L)
            for j in range(L):
                line.GetPointIds().SetId(j, i * L + j)
            lines.InsertNextCell(line)
        trail = vtk.vtkPolyData(); trail.SetPoints(tp); trail.SetLines(lines)
        sc = numpy_to_vtk(np.repeat(d["tracer_speed"], L), deep=True); sc.SetName("speed")
        trail.GetPointData().SetScalars(sc)
        tm = vtk.vtkPolyDataMapper(); tm.SetInputData(trail)
        tm.SetLookupTable(lut); tm.SetScalarRange(0, VMAX); tm.UseLookupTableScalarRangeOn()
        ta = vtk.vtkActor(); ta.SetMapper(tm); ta.GetProperty().SetLineWidth(2.0); ta.GetProperty().SetOpacity(0.7)
        ren.AddActor(ta)

    # faint speed slice in the beat plane x3 = x0_3
    plane = vtk.vtkPlane(); plane.SetOrigin(*x0); plane.SetNormal(0, 0, 1)
    cut = vtk.vtkCutter(); cut.SetInputData(grid); cut.SetCutFunction(plane)
    cm = vtk.vtkPolyDataMapper(); cm.SetInputConnection(cut.GetOutputPort())
    cm.SetScalarModeToUsePointFieldData(); cm.SelectColorArray("speed"); cm.SetLookupTable(lut)
    cm.SetScalarRange(0, VMAX); cm.UseLookupTableScalarRangeOn()
    ca = vtk.vtkActor(); ca.SetMapper(cm); ca.GetProperty().SetOpacity(0.22); ca.GetProperty().LightingOff()
    ren.AddActor(ca)

    # cell body: ellipsoid oriented by B
    sph = vtk.vtkParametricEllipsoid(); sph.SetXRadius(ax[0]); sph.SetYRadius(ax[1]); sph.SetZRadius(ax[2])
    src = vtk.vtkParametricFunctionSource(); src.SetParametricFunction(sph); src.SetUResolution(60); src.SetVResolution(60)
    M = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            M.SetElement(i, j, B[i, j])
        M.SetElement(i, 3, x0[i])
    hm = vtk.vtkPolyDataMapper(); hm.SetInputConnection(src.GetOutputPort()); hm.ScalarVisibilityOff()
    ha = vtk.vtkActor(); ha.SetMapper(hm); ha.SetUserMatrix(M)
    ha.GetProperty().SetColor(0.30, 0.68, 0.31); ha.GetProperty().SetSpecular(0.4); ha.GetProperty().SetSpecularPower(30)
    ren.AddActor(ha)

    # flagella as tubes (radius = eps = 0.0125, drawn a bit thicker to be visible)
    for P in (d["cis"], d["trans"]):
        ren.AddActor(tube_actor(polyline(P), 0.022, (0.12, 0.40, 0.14)))

    # path of the cell centre so far
    path = np.array([np.load(g)["x0"] for g in all_files[: k + 1]])
    if len(path) > 1:
        ren.AddActor(tube_actor(polyline(path), 0.01, (0.85, 0.2, 0.2)))

    # scalar bar
    bar = vtk.vtkScalarBarActor(); bar.SetLookupTable(lut); bar.SetTitle("flow speed"); bar.SetNumberOfLabels(3)
    bar.SetWidth(0.07); bar.SetHeight(0.35); bar.SetPosition(0.9, 0.08)
    bar.GetTitleTextProperty().SetFontSize(14); bar.GetLabelTextProperty().SetFontSize(12)
    ren.AddActor2D(bar)

    txt = vtk.vtkTextActor()
    txt.SetInput(f"Model biflagellate (Gallagher & Smith 2018)   beat {d['t'] / (2 * math.pi):.2f}   "
                 f"swum {x0[1] - d['start'][1]:+.3f} flagellar lengths")
    txt.GetTextProperty().SetFontSize(18); txt.GetTextProperty().SetColor(0.95, 0.95, 0.95)
    txt.SetDisplayPosition(20, 725)
    ren.AddActor2D(txt)

    # camera: oblique view, orbiting slowly around the swimming (x2) axis
    cam = ren.GetActiveCamera()
    az = math.radians(-50 + 100 * k / 60)       # orbit about the swimming axis x2
    el = math.radians(28)                        # look down at the cell from behind and above
    c = x0 + [0, 0.3, 0]
    dist = 6.2
    cam.SetFocalPoint(*c)
    cam.SetPosition(c[0] + dist * math.sin(az) * math.cos(el), c[1] - dist * math.sin(el),
                    c[2] + dist * math.cos(az) * math.cos(el))
    cam.SetViewUp(0, 1, 0)
    cam.SetViewAngle(30)
    ren.ResetCameraClippingRange()

    win.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
    png = vtk.vtkPNGWriter(); png.SetFileName(f"{folder}/img{k:03d}.png"); png.SetInputConnection(w2i.GetOutputPort()); png.Write()
print("rendered", len(files))
