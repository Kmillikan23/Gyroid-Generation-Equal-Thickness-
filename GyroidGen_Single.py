import os
from GenFunctions.Gyroid_Thicken import NotCorrectedThick
from GenFunctions.Gyroid_Isosurface import GyroidIso

from types import SimpleNamespace
import pyvista as pv
from scipy.optimize import root_scalar


length = (10 10 10)   # mm
res    = 0.23          # mm

wavex, wavey, wavez = 10,10,10     # mm
thetax, thetay, thetaz = 0.0, 0.0, 0.0    # degrees
target_porosity = 0.7


output_dir = "output_STLs"
os.makedirs(output_dir, exist_ok=True)

# single structure params
params = SimpleNamespace(
    length      = length,
    wavelengths = (wavex, wavey, wavez),
    rotation    = (thetax, thetay, thetaz),
    res         = res,
    thickness   = None
)

def compute_porosity(thk):
    params.thickness = thk
    iso   = GyroidIso(params, 0)
    mesh  = NotCorrectedThick(params, iso).connectivity(extraction_mode="largest")
    vol   = mesh.volume
    por   = 1.0 - vol / (length[0]*length[1]*length[2])
    return por, mesh

# root-finding function
f = lambda t: compute_porosity(t)[0] - target_porosity

a, b = 0.2, 0.8
fa, fb = f(a), f(b)
while fa * fb > 0:
    b *= 1.25
    fb = f(b)

sol = root_scalar(f, bracket=[a, b], method='brentq', xtol=1e-2)
thickness_opt = sol.root

# compute final mesh & save
_, mesh_opt = compute_porosity(thickness_opt)
filename = "Gyroid_output.stl"
stlsave = os.path.join(output_dir, filename)
mesh_opt.save(stlsave)

plotter = pv.Plotter()
plotter.add_mesh(mesh_opt, color='lightblue', show_edges=True)
#plotter.show_grid()
plotter.show()

print(f"Done. thickness = {thickness_opt:.4f} mm  |  STL: {stlsave}")

